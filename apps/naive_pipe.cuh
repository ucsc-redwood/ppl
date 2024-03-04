#pragma once

#include "cuda/agents/prefix_sum_agent.cuh"
#include "cuda/unified_vector.cuh"

// ok let's list out some memories we need

// Essential:
// 1. u_points (n)
// 2. u_morton_keys (n)
//    - u_sort_alt
//    - u_global_histogram
//    - u_index
//    - u_pass_histogram (x4)
// 3. u_unique_keys (n_unique)
//    - u_flag_heads
//    - u_aux (optional)
// 4. Radix Node data...
// 5. u_edge_count
// 6. u_edge_offset
//     - u_auxlitary
// 7. Octree Node data...

struct NaivePipe {
  static constexpr auto educated_guess_nodes = 0.6f;

  int n_pts;
  int n_unique_keys;
  int n_brt_nodes;  // unique_keys - 1
  int n_oct_nodes;  // computed late... we use 0.6 * n as a guess

  // Essential data memory
  cu::unified_vector<glm::vec4> u_points;
  cu::unified_vector<unsigned int> u_morton_keys;
  cu::unified_vector<unsigned int> u_unique_morton_keys;
  cu::unified_vector<int> u_edge_count;
  cu::unified_vector<int> u_edge_offset;

  // Essential
  // should be of size 'n_unique_keys - 1', but we can allocate as 'n' for now
  struct {
    cu::unified_vector<uint8_t> u_prefix_n;
    bool* u_has_leaf_left;  // you can't use vector of bools
    bool* u_has_leaf_right;
    cu::unified_vector<int> u_left_child;
    cu::unified_vector<int> u_parent;
  } brt;

  // Essential
  // should be of size 'n_oct_nodes', we use an educated guess for now
  struct {
    int (*u_children)[8];
    glm::vec4* u_corner;
    float* u_cell_size;
    int* u_child_node_mask;
    int* u_child_leaf_mask;
  } oct;

  // Temp
  struct {
    cu::unified_vector<unsigned int> u_sort_alt;              // n
    cu::unified_vector<unsigned int> u_global_histogram;      // 256 * 4
    cu::unified_vector<unsigned int> u_index;                 // 4
    cu::unified_vector<unsigned int> u_first_pass_histogram;  // 256 * xxx
    cu::unified_vector<unsigned int> u_second_pass_histogram;
    cu::unified_vector<unsigned int> u_third_pass_histogram;
    cu::unified_vector<unsigned int> u_fourth_pass_histogram;
  } sort_tmp;

  struct {
    cu::unified_vector<int> u_flag_heads;  // n
  } unique_tmp;

  struct {
    // use Agent's tile size to allocate
    cu::unified_vector<int> u_auxiliary;  // n_tiles
  } prefix_sum_tmp;

  explicit NaivePipe(const int n) : n_pts(n) {
    // --- Essentials ---
    u_points.resize(n);
    u_morton_keys.resize(n);
    u_unique_morton_keys.resize(n);

    brt.u_prefix_n.resize(n);  // should be n_unique, but n will do for now
    MallocManaged(&brt.u_has_leaf_left, n);
    MallocManaged(&brt.u_has_leaf_right, n);
    brt.u_left_child.resize(n);
    brt.u_parent.resize(n);

    u_edge_count.resize(n);
    u_edge_offset.resize(n);

    const auto num_oct_to_allocate = n * educated_guess_nodes;
    MallocManaged(&oct.u_children, num_oct_to_allocate);
    MallocManaged(&oct.u_corner, num_oct_to_allocate);
    MallocManaged(&oct.u_cell_size, num_oct_to_allocate);
    MallocManaged(&oct.u_child_node_mask, num_oct_to_allocate);
    MallocManaged(&oct.u_child_leaf_mask, num_oct_to_allocate);

    // -------------------------

    // Temporary storages for Sort
    constexpr auto radix = 256;
    constexpr auto passes = 4;
    const auto binning_thread_blocks = cub::DivideAndRoundUp(n, 7680);
    sort_tmp.u_sort_alt.resize(n);
    sort_tmp.u_global_histogram.resize(radix * passes);
    sort_tmp.u_index.resize(passes);
    sort_tmp.u_first_pass_histogram.resize(radix * binning_thread_blocks);
    sort_tmp.u_second_pass_histogram.resize(radix * binning_thread_blocks);
    sort_tmp.u_third_pass_histogram.resize(radix * binning_thread_blocks);
    sort_tmp.u_fourth_pass_histogram.resize(radix * binning_thread_blocks);

    // Temporary storages for Unique
    unique_tmp.u_flag_heads.resize(n);

    // Temporary storages for PrefixSum
    constexpr auto prefix_sum_tile_size = gpu::PrefixSumAgent<int>::tile_size;
    const auto prefix_sum_n_tiles =
        cub::DivideAndRoundUp(n, prefix_sum_tile_size);
    prefix_sum_tmp.u_auxiliary.resize(prefix_sum_n_tiles);
  }

  ~NaivePipe() {
    CUDA_FREE(brt.u_has_leaf_left);
    CUDA_FREE(brt.u_has_leaf_right);

    CUDA_FREE(oct.u_children);
    CUDA_FREE(oct.u_corner);
    CUDA_FREE(oct.u_cell_size);
    CUDA_FREE(oct.u_child_node_mask);
    CUDA_FREE(oct.u_child_leaf_mask);
  }

  void attachStream(const cudaStream_t stream) {
    ATTACH_STREAM_SINGLE(u_points.data());
    ATTACH_STREAM_SINGLE(u_morton_keys.data());
    ATTACH_STREAM_SINGLE(u_unique_morton_keys.data());

    ATTACH_STREAM_SINGLE(sort_tmp.u_sort_alt.data());
    ATTACH_STREAM_SINGLE(sort_tmp.u_global_histogram.data());
    ATTACH_STREAM_SINGLE(sort_tmp.u_index.data());
    ATTACH_STREAM_SINGLE(sort_tmp.u_first_pass_histogram.data());
    ATTACH_STREAM_SINGLE(sort_tmp.u_second_pass_histogram.data());
    ATTACH_STREAM_SINGLE(sort_tmp.u_third_pass_histogram.data());
    ATTACH_STREAM_SINGLE(sort_tmp.u_fourth_pass_histogram.data());

    ATTACH_STREAM_SINGLE(unique_tmp.u_flag_heads.data());

    ATTACH_STREAM_SINGLE(brt.u_prefix_n.data());
    ATTACH_STREAM_SINGLE(brt.u_has_leaf_left);
    ATTACH_STREAM_SINGLE(brt.u_has_leaf_right);
    ATTACH_STREAM_SINGLE(brt.u_left_child.data());
    ATTACH_STREAM_SINGLE(brt.u_parent.data());

    ATTACH_STREAM_SINGLE(u_edge_count.data());
    ATTACH_STREAM_SINGLE(u_edge_offset.data());

    ATTACH_STREAM_SINGLE(prefix_sum_tmp.u_auxiliary.data());

    ATTACH_STREAM_SINGLE(oct.u_children);
    ATTACH_STREAM_SINGLE(oct.u_corner);
    ATTACH_STREAM_SINGLE(oct.u_cell_size);
    ATTACH_STREAM_SINGLE(oct.u_child_node_mask);
    ATTACH_STREAM_SINGLE(oct.u_child_leaf_mask);
  }
};
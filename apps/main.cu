#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cub/cub.cuh>
#include <iostream>
#include <memory>

#include "app_params.hpp"
#include "types/pipe.cuh"

//
#include "cuda/agents/prefix_sum_agent.cuh"
#include "cuda/dispatchers/edge_count_dispatch.cuh"
#include "cuda/dispatchers/init_dispatch.cuh"
#include "cuda/dispatchers/morton_dispatch.cuh"
#include "cuda/dispatchers/prefix_sum_dispatch.cuh"
#include "cuda/dispatchers/radix_tree_dispatch.cuh"
#include "cuda/dispatchers/sort_dispatch.cuh"
#include "cuda/dispatchers/unique_dispatch.cuh"

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

  cu::unified_vector<glm::vec4> u_points;
  cu::unified_vector<unsigned int> u_morton_keys;
  cu::unified_vector<unsigned int> u_unique_morton_keys;
  cu::unified_vector<int> u_edge_count;
  cu::unified_vector<int> u_edge_offset;

  // should be of size 'n_unique_keys - 1', but we can allocate as 'n' for now
  struct {
    cu::unified_vector<uint8_t> u_prefix_n;
    bool* u_has_leaf_left;  // you can't use vector of bools
    bool* u_has_leaf_right;
    cu::unified_vector<int> u_left_child;
    cu::unified_vector<int> u_parent;
  } brt;

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
    // constexpr auto tile_size = gpu::PrefixSumAgent<int>::tile_size;
    // const auto n_tiles = cub::DivideAndRoundUp(n, tile_size);
    // cu::unified_vector<int> u_auxiliary(n_tiles);
    cu::unified_vector<int> u_auxiliary;  // n_tiles
  } prefix_sum_tmp;

  explicit NaivePipe(const int n) : n_pts(n) {
    // Essential
    u_points.resize(n);
    u_morton_keys.resize(n);
    u_unique_morton_keys.resize(n);

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

    // BRT
    brt.u_prefix_n.resize(n);  // should be n_unique, but n will do for now
    MallocManaged(&brt.u_has_leaf_left, n * sizeof(bool));
    MallocManaged(&brt.u_has_leaf_right, n * sizeof(bool));
    brt.u_left_child.resize(n);
    brt.u_parent.resize(n);

    // Edge count
    u_edge_count.resize(n);
    u_edge_offset.resize(n);

    // Temporary storages for PrefixSum
    constexpr auto prefix_sum_tile_size = gpu::PrefixSumAgent<int>::tile_size;
    const auto prefix_sum_n_tiles =
        cub::DivideAndRoundUp(n, prefix_sum_tile_size);
    prefix_sum_tmp.u_auxiliary.resize(prefix_sum_n_tiles);
  }

  ~NaivePipe() {
    CUDA_FREE(brt.u_has_leaf_left);
    CUDA_FREE(brt.u_has_leaf_right);
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
  }
};

int main(const int argc, const char** argv) {
  AppParams params(argc, argv);
  params.print_params();

  spdlog::set_level(params.debug_print ? spdlog::level::debug
                                       : spdlog::level::info);

  // ------------------------------
  constexpr auto n_streams = 1;

  std::array<cudaStream_t, n_streams> streams;
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  }

  const auto pipe = std::make_unique<NaivePipe>(params.n);
  {
    gpu::dispatch_InitRandomVec4(params.n_blocks,
                                 streams[0],
                                 pipe->u_points.data(),
                                 params.n,
                                 params.min_coord,
                                 params.getRange(),
                                 params.seed);

    gpu::dispatch_ComputeMorton(params.n_blocks,
                                streams[0],
                                pipe->u_points.data(),
                                pipe->u_morton_keys.data(),
                                params.n,
                                params.min_coord,
                                params.getRange());

    gpu::dispatch_RadixSort(params.n_blocks,
                            streams[0],
                            pipe->u_morton_keys.data(),
                            pipe->sort_tmp.u_sort_alt.data(),
                            pipe->sort_tmp.u_global_histogram.data(),
                            pipe->sort_tmp.u_index.data(),
                            pipe->sort_tmp.u_first_pass_histogram.data(),
                            pipe->sort_tmp.u_second_pass_histogram.data(),
                            pipe->sort_tmp.u_third_pass_histogram.data(),
                            pipe->sort_tmp.u_fourth_pass_histogram.data(),
                            params.n);
    int num_unique;
    gpu::dispatch_Unique_easy(params.n_blocks,
                              streams[0],
                              pipe->u_morton_keys.data(),
                              pipe->u_unique_morton_keys.data(),
                              pipe->unique_tmp.u_flag_heads.data(),
                              params.n,
                              num_unique);

    SYNC_STREAM(streams[0]);

    spdlog::info("num_unique: {}/{}", num_unique, params.n);
    pipe->n_unique_keys = num_unique;
    pipe->n_brt_nodes = num_unique - 1;

    gpu::dispatch_BuildRadixTree(params.n_blocks,
                                 streams[0],
                                 pipe->u_unique_morton_keys.data(),
                                 pipe->brt.u_prefix_n.data(),
                                 pipe->brt.u_has_leaf_left,
                                 pipe->brt.u_has_leaf_right,
                                 pipe->brt.u_left_child.data(),
                                 pipe->brt.u_parent.data(),
                                 pipe->n_unique_keys);

    gpu::dispatch_EdgeCount(params.n_blocks,
                            streams[0],
                            pipe->brt.u_prefix_n.data(),
                            pipe->brt.u_parent.data(),
                            pipe->u_edge_count.data(),
                            pipe->n_brt_nodes);

    gpu::dispatch_PrefixSum(params.n_blocks,
                            streams[0],
                            pipe->u_edge_count.data(),
                            pipe->u_edge_offset.data(),
                            pipe->prefix_sum_tmp.u_auxiliary.data(),
                            pipe->n_brt_nodes);

    SYNC_STREAM(streams[0]);
  }

  // ------------------------------
  // peek 32 brt nodes
  spdlog::trace("Peeking 32 BRT nodes...");

  for (int i = 0; i < 32; i++) {
    spdlog::trace(
        "Node {}: prefix_n: {}, has_leaf_left: {}, has_leaf_right: {}, "
        "left_child: {}, parent: {}",
        i,
        pipe->brt.u_prefix_n[i],
        pipe->brt.u_has_leaf_left[i],
        pipe->brt.u_has_leaf_right[i],
        pipe->brt.u_left_child[i],
        pipe->brt.u_parent[i]);
  }

  // peek 32 edge counts
  for (int i = 0; i < 32; i++) {
    spdlog::debug("Node {}: u_edge_offset: {}", i, pipe->u_edge_offset[i]);
  }

  spdlog::info("Done");
  for (const auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  }
  return 0;
}
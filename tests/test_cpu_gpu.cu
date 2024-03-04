#include <gtest/gtest.h>

#include "cuda/unified_vector.cuh"
#include "kernels_fwd.h"

#define CREATE_STREAM  \
  cudaStream_t stream; \
  CHECK_CUDA_CALL(cudaStreamCreate(&stream));

#define DESCROY_STREAM CHECK_CUDA_CALL(cudaStreamDestroy(stream));

// ----------------------------------------------------------------------------
// Test Morton
// ----------------------------------------------------------------------------

static void Test_ComputeMorton(const int n, const int n_gpu_blocks) {
  CREATE_STREAM;

  cu::unified_vector<glm::vec4> u_points(n);
  std::generate(u_points.begin(), u_points.end(), [i = 0]() mutable {
    return glm::vec4(i, i, i, i++);
  });

  cu::unified_vector<unsigned int> cpu_result(n);
  cu::unified_vector<unsigned int> gpu_result(n);

  // ------

  gpu::dispatch_ComputeMorton(
      n_gpu_blocks, stream, u_points.data(), gpu_result.data(), n, 0, n);
  SYNC_DEVICE();

  cpu::k_ComputeMortonCode(u_points.data(), cpu_result.data(), n, 0, n);

  // ------

  for (auto i = 0; i < n; ++i) {
    EXPECT_EQ(cpu_result[i], gpu_result[i]);
  }

  DESCROY_STREAM;
}

TEST(Test_ComputeMorton, RegularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_ComputeMorton(1 << 12, 1));  // 1 << 12 = 4096
  EXPECT_NO_FATAL_FAILURE(Test_ComputeMorton(1 << 14, 2));
  EXPECT_NO_FATAL_FAILURE(Test_ComputeMorton(1 << 16, 4));
  EXPECT_NO_FATAL_FAILURE(Test_ComputeMorton(1 << 18, 8));
  EXPECT_NO_FATAL_FAILURE(Test_ComputeMorton(1 << 20, 16));
}

TEST(Test_ComputeMorton, IrregularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_ComputeMorton(114514, 1));
  EXPECT_NO_FATAL_FAILURE(Test_ComputeMorton(640 * 480, 4));
  EXPECT_NO_FATAL_FAILURE(Test_ComputeMorton(1920 * 1080, 16));
}

// ----------------------------------------------------------------------------
// Skipping Sorting, as it is already tested in test_sort.cu
// Test Radix Tree
// ----------------------------------------------------------------------------

struct RadixTree {
  explicit RadixTree(const int n, cudaStream_t& stream)
      : u_prefix_n(n), u_left_child(n), u_parent(n) {
    MallocManaged(&u_has_leaf_left, n);
    MallocManaged(&u_has_leaf_right, n);

    ATTACH_STREAM_SINGLE(u_prefix_n.data());
    ATTACH_STREAM_SINGLE(u_has_leaf_left);
    ATTACH_STREAM_SINGLE(u_has_leaf_right);
    ATTACH_STREAM_SINGLE(u_left_child.data());
    ATTACH_STREAM_SINGLE(u_parent.data());
  }

  ~RadixTree() {
    CUDA_FREE(u_has_leaf_left);
    CUDA_FREE(u_has_leaf_right);
  }

  cu::unified_vector<uint8_t> u_prefix_n;
  bool* u_has_leaf_left;
  bool* u_has_leaf_right;
  cu::unified_vector<int> u_left_child;
  cu::unified_vector<int> u_parent;
};

static void Test_RadixTree(const int n, const int n_gpu_blocks) {
  CREATE_STREAM;

  cu::unified_vector<unsigned int> u_unique_morton_keys(n);
  std::iota(u_unique_morton_keys.begin(), u_unique_morton_keys.end(), 0);

  const auto n_unique = n;
  const auto n_brt_nodes = n - 1;

  RadixTree cpu_brt(n_brt_nodes, stream);
  RadixTree gpu_brt(n_brt_nodes, stream);

  // ------

  gpu::dispatch_BuildRadixTree(n_gpu_blocks,
                               stream,
                               u_unique_morton_keys.data(),
                               gpu_brt.u_prefix_n.data(),
                               gpu_brt.u_has_leaf_left,
                               gpu_brt.u_has_leaf_right,
                               gpu_brt.u_left_child.data(),
                               gpu_brt.u_parent.data(),
                               n_unique);
  SYNC_DEVICE();

  cpu::k_BuildRadixTree(n_unique,
                        u_unique_morton_keys.data(),
                        cpu_brt.u_prefix_n.data(),
                        cpu_brt.u_has_leaf_left,
                        cpu_brt.u_has_leaf_right,
                        cpu_brt.u_left_child.data(),
                        cpu_brt.u_parent.data());

  // ------

  auto is_equal = std::equal(gpu_brt.u_prefix_n.begin(),
                             gpu_brt.u_prefix_n.end(),
                             cpu_brt.u_prefix_n.begin());
  EXPECT_TRUE(is_equal);

  is_equal = std::equal(gpu_brt.u_has_leaf_left,
                        gpu_brt.u_has_leaf_left + n_brt_nodes,
                        cpu_brt.u_has_leaf_left);
  EXPECT_TRUE(is_equal);

  is_equal = std::equal(gpu_brt.u_has_leaf_right,
                        gpu_brt.u_has_leaf_right + n_brt_nodes,
                        cpu_brt.u_has_leaf_right);
  EXPECT_TRUE(is_equal);

  is_equal = std::equal(gpu_brt.u_left_child.begin(),
                        gpu_brt.u_left_child.end(),
                        cpu_brt.u_left_child.begin());
  EXPECT_TRUE(is_equal);

  is_equal = std::equal(gpu_brt.u_parent.begin(),
                        gpu_brt.u_parent.end(),
                        cpu_brt.u_parent.begin());
  EXPECT_TRUE(is_equal);

  DESCROY_STREAM;
}

TEST(Test_RadixTree, RegularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_RadixTree(1 << 12, 1));  // 1 << 12 = 4096
  EXPECT_NO_FATAL_FAILURE(Test_RadixTree(1 << 14, 2));
  EXPECT_NO_FATAL_FAILURE(Test_RadixTree(1 << 16, 4));
  EXPECT_NO_FATAL_FAILURE(Test_RadixTree(1 << 18, 8));
  EXPECT_NO_FATAL_FAILURE(Test_RadixTree(1 << 20, 16));
}

TEST(Test_RadixTree, IrregularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_RadixTree(114514, 1));
  EXPECT_NO_FATAL_FAILURE(Test_RadixTree(640 * 480, 4));
  EXPECT_NO_FATAL_FAILURE(Test_RadixTree(1920 * 1080, 16));
}

// ----------------------------------------------------------------------------
// Testing Edge Count
// ----------------------------------------------------------------------------

static void Test_EdgeCount(const int n, const int n_gpu_blocks) {
  CREATE_STREAM;

  cu::unified_vector<unsigned int> u_unique_morton_keys(n);
  std::iota(u_unique_morton_keys.begin(), u_unique_morton_keys.end(), 0);

  const auto n_unique = n;
  const auto n_brt_nodes = n - 1;

  RadixTree input_brt(n_brt_nodes, stream);
  cpu::k_BuildRadixTree(n_unique,
                        u_unique_morton_keys.data(),
                        input_brt.u_prefix_n.data(),
                        input_brt.u_has_leaf_left,
                        input_brt.u_has_leaf_right,
                        input_brt.u_left_child.data(),
                        input_brt.u_parent.data());

  // ------

  cu::unified_vector<int> cpu_edge_count(n_brt_nodes);
  cu::unified_vector<int> gpu_edge_count(n_brt_nodes);

  gpu::dispatch_EdgeCount(n_gpu_blocks,
                          stream,
                          input_brt.u_prefix_n.data(),
                          input_brt.u_parent.data(),
                          gpu_edge_count.data(),
                          n_brt_nodes);
  SYNC_DEVICE();

  cpu::k_EdgeCount(input_brt.u_prefix_n.data(),
                   input_brt.u_parent.data(),
                   cpu_edge_count.data(),
                   n_brt_nodes);

  // ------

  auto is_equal = std::equal(
      gpu_edge_count.begin(), gpu_edge_count.end(), cpu_edge_count.begin());
  EXPECT_TRUE(is_equal);

  DESCROY_STREAM;
}

TEST(Test_EdgeCount, RegularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_EdgeCount(1 << 12, 1));  // 1 << 12 = 4096
  EXPECT_NO_FATAL_FAILURE(Test_EdgeCount(1 << 14, 2));
  EXPECT_NO_FATAL_FAILURE(Test_EdgeCount(1 << 16, 4));
  EXPECT_NO_FATAL_FAILURE(Test_EdgeCount(1 << 18, 8));
  EXPECT_NO_FATAL_FAILURE(Test_EdgeCount(1 << 20, 16));
}

TEST(Test_EdgeCount, IrregularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_EdgeCount(114514, 1));
  EXPECT_NO_FATAL_FAILURE(Test_EdgeCount(640 * 480, 4));
  EXPECT_NO_FATAL_FAILURE(Test_EdgeCount(1920 * 1080, 16));
}

// ----------------------------------------------------------------------------
// Testing Prefix Sum
// ----------------------------------------------------------------------------

static void Test_PrefixSum(const int n, const int n_gpu_blocks) {
  CREATE_STREAM;

  cu::unified_vector<int> u_data(n);
  std::iota(u_data.begin(), u_data.end(), 0);

  cu::unified_vector<int> cpu_output(n);
  cu::unified_vector<int> gpu_output(n);

  constexpr auto tile_size = gpu::PrefixSumAgent<int>::tile_size;
  const auto n_tiles = cub::DivideAndRoundUp(n, tile_size);
  cu::unified_vector<int> u_auxiliary(n_tiles);

  // ------

  gpu::dispatch_PrefixSum(n_gpu_blocks,
                          stream,
                          u_data.data(),
                          gpu_output.data(),
                          u_auxiliary.data(),
                          n);
  SYNC_DEVICE();

  cpu::std_exclusive_scan(u_data.data(), cpu_output.data(), n);

  // ------

  auto is_equal =
      std::equal(gpu_output.begin(), gpu_output.end(), cpu_output.begin());
  EXPECT_TRUE(is_equal);

  DESCROY_STREAM;
}

TEST(Test_PrefixSum, RegularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1 << 12, 1));  // 1 << 12 = 4096
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1 << 14, 2));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1 << 16, 4));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1 << 18, 8));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1 << 20, 16));
}

TEST(Test_PrefixSum, IrregularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(114514, 1));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(640 * 480, 4));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1920 * 1080, 16));
}

// ----------------------------------------------------------------------------
// Testing Make Octree
// ----------------------------------------------------------------------------

struct Octree {
  explicit Octree(const int n, cudaStream_t& stream) {
    MallocManaged(&u_children, n * 8);
    MallocManaged(&u_corner, n);
    MallocManaged(&u_cell_size, n);
    MallocManaged(&u_child_node_mask, n);
    MallocManaged(&u_child_leaf_mask, n);

    ATTACH_STREAM_SINGLE(u_children);
    ATTACH_STREAM_SINGLE(u_corner);
    ATTACH_STREAM_SINGLE(u_cell_size);
    ATTACH_STREAM_SINGLE(u_child_node_mask);
    ATTACH_STREAM_SINGLE(u_child_leaf_mask);
  }

  ~Octree() {
    CUDA_FREE(u_children);
    CUDA_FREE(u_corner);
    CUDA_FREE(u_cell_size);
    CUDA_FREE(u_child_node_mask);
    CUDA_FREE(u_child_leaf_mask);
  }

  int (*u_children)[8];
  glm::vec4* u_corner;
  float* u_cell_size;
  int* u_child_node_mask;
  int* u_child_leaf_mask;
};

static void Test_MakeOctreeNodes(const int n, const int n_gpu_blocks) {
  CREATE_STREAM;

  cu::unified_vector<unsigned int> u_unique_morton_keys(n);
  std::iota(u_unique_morton_keys.begin(), u_unique_morton_keys.end(), 0);

  const auto n_unique = n;
  const auto n_brt_nodes = n - 1;

  RadixTree input_brt(n_brt_nodes, stream);

  cpu::k_BuildRadixTree(n_unique,
                        u_unique_morton_keys.data(),
                        input_brt.u_prefix_n.data(),
                        input_brt.u_has_leaf_left,
                        input_brt.u_has_leaf_right,
                        input_brt.u_left_child.data(),
                        input_brt.u_parent.data());

  cu::unified_vector<int> u_edge_count(n_brt_nodes);
  cu::unified_vector<int> u_edge_offset(n_brt_nodes + 1);

  cpu::k_EdgeCount(input_brt.u_prefix_n.data(),
                   input_brt.u_parent.data(),
                   u_edge_count.data(),
                   n_brt_nodes);

  cpu::std_exclusive_scan(
      u_edge_count.data(), u_edge_offset.data(), n_brt_nodes);

  // prepare outputs

  Octree cpu_oct(n_brt_nodes, stream);
  Octree gpu_oct(n_brt_nodes, stream);

  // ------

  gpu::dispatch_BuildOctree(n_gpu_blocks,
                            stream,
                            gpu_oct.u_children,
                            gpu_oct.u_corner,
                            gpu_oct.u_cell_size,
                            gpu_oct.u_child_node_mask,
                            gpu_oct.u_child_leaf_mask,
                            u_edge_offset.data(),
                            u_edge_count.data(),
                            u_unique_morton_keys.data(),
                            input_brt.u_prefix_n.data(),
                            input_brt.u_has_leaf_left,
                            input_brt.u_has_leaf_right,
                            input_brt.u_left_child.data(),
                            input_brt.u_parent.data(),
                            0,
                            n,
                            n_brt_nodes);
  SYNC_DEVICE();

  cpu::k_MakeOctNodes(cpu_oct.u_children,
                      cpu_oct.u_corner,
                      cpu_oct.u_cell_size,
                      cpu_oct.u_child_node_mask,
                      u_edge_offset.data(),
                      u_edge_count.data(),
                      u_unique_morton_keys.data(),
                      input_brt.u_prefix_n.data(),
                      input_brt.u_parent.data(),
                      0,
                      n,
                      n_brt_nodes);

  // ------

  auto is_equal = std::equal(gpu_oct.u_children[0],
                             gpu_oct.u_children[0] + n_brt_nodes * 8,
                             cpu_oct.u_children[0]);
  EXPECT_TRUE(is_equal);

  DESCROY_STREAM;
}

TEST(BuildOctree, Test_MakeOctreeNodes) {
  EXPECT_NO_FATAL_FAILURE(Test_MakeOctreeNodes(1 << 12, 1));  // 1 << 12 = 4096
  EXPECT_NO_FATAL_FAILURE(Test_MakeOctreeNodes(1 << 14, 2));
  EXPECT_NO_FATAL_FAILURE(Test_MakeOctreeNodes(1 << 16, 4));
  EXPECT_NO_FATAL_FAILURE(Test_MakeOctreeNodes(1 << 18, 8));
  EXPECT_NO_FATAL_FAILURE(Test_MakeOctreeNodes(1 << 20, 16));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#include <gtest/gtest.h>

#include <cub/cub.cuh>
#include <numeric>
#include <vector>

#include "cuda/unified_vector.cuh"

template __global__ void k_PrefixSumLocal(const unsigned int* u_input,
                                          unsigned int* u_output,
                                          int n,
                                          unsigned int* u_auxiliary);

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

static void Test_PrefixSumLocal(const int n, const int n_blocks) {
  const cu::unified_vector<unsigned int> u_data(n, 1);
  cu::unified_vector<unsigned int> u_output(n);

  constexpr auto n_threads = PrefixSumAgent<unsigned int>::n_threads;

  k_PrefixSumLocal<<<n_blocks, n_threads>>>(u_data.data(), u_output.data(), n);
  SYNC_DEVICE();

  constexpr auto items_per_thread =
      PrefixSumAgent<unsigned int>::items_per_thread;
  constexpr auto tile_size = n_threads * items_per_thread;
  const auto n_tiles = (n + tile_size - 1) / tile_size;

  for (auto tile_id = 0; tile_id < n_tiles; ++tile_id) {
    for (auto i = 0; i < tile_size; ++i) {
      if (const auto idx = tile_id * tile_size + i; idx < n) {
        EXPECT_EQ(u_output[idx], i);
      }
    }
  }
}

// Test case
TEST(PrefixSumTestRegular, Test_PrefixSumLocal) {
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(1 << 10, 1));  // 1024
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(1 << 16, 1));  // 65536
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(1 << 20, 1));  // 1048576
}

TEST(PrefixSumTestRegularMultipleBlocks, Test_PrefixSumLocal) {
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(1 << 10, 1));   // 1024
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(1 << 16, 2));   // 65536
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(1 << 20, 4));   // 1048576
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(1 << 20, 8));   // 1048576
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(1 << 20, 16));  // 1048576
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(1 << 20, 32));  // 1048576
}

TEST(PrefixSumTestIrregular, Test_PrefixSumLocal) {
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(1234, 1));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(114514, 2));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(1919810, 3));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(810893, 4));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumLocal(810893, 5));
}

static void Test_SingleBlockPrefixSum(const int n) {
  cu::unified_vector<unsigned int> u_data(n, 1);
  cu::unified_vector<unsigned int> u_output(n);

  constexpr auto n_threads = PrefixSumAgent<unsigned int>::n_threads;

  k_SingleBlockExclusiveScan<<<1, n_threads>>>(
      u_data.data(), u_output.data(), n);
  SYNC_DEVICE();

  std::vector<unsigned int> cpu_output(n);
  std::exclusive_scan(u_data.begin(), u_data.end(), cpu_output.begin(), 0);

  const auto is_equal =
      std::equal(u_output.begin(), u_output.end(), cpu_output.begin());
  EXPECT_TRUE(is_equal);
}

static void Test_SingleBlockPrefixSum_SelfToSelf(const int n) {
  cu::unified_vector<unsigned int> u_data(n, 1);
  std::vector<unsigned int> cpu_output(n);
  std::exclusive_scan(u_data.begin(), u_data.end(), cpu_output.begin(), 0);

  constexpr auto n_threads = PrefixSumAgent<unsigned int>::n_threads;

  k_SingleBlockExclusiveScan<<<1, n_threads>>>(u_data.data(), u_data.data(), n);
  SYNC_DEVICE();

  const auto is_equal =
      std::equal(u_data.begin(), u_data.end(), cpu_output.begin());
  EXPECT_TRUE(is_equal);
}

// Test case
TEST(PrefixSumTestRegular, Test_SingleBlockPrefixSum) {
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlockPrefixSum(1 << 10));  // 1024
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlockPrefixSum(1 << 16));  // 65536
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlockPrefixSum(1 << 20));  // 1048576
}

TEST(PrefixSumTestRegularSelfToSelf, Test_SingleBlockPrefixSum) {
  EXPECT_NO_FATAL_FAILURE(
      Test_SingleBlockPrefixSum_SelfToSelf(1 << 10));  // 1024
  EXPECT_NO_FATAL_FAILURE(
      Test_SingleBlockPrefixSum_SelfToSelf(1 << 16));  // 65536
  EXPECT_NO_FATAL_FAILURE(
      Test_SingleBlockPrefixSum_SelfToSelf(1 << 20));  // 1048576
}

static void Test_GlobalPrefixSum(const int n, const int n_blocks) {
  const cu::unified_vector<unsigned int> u_data(n, 1);
  cu::unified_vector<unsigned int> u_local_sums(n);

  constexpr auto n_threads = PrefixSumAgent<unsigned int>::n_threads;
  constexpr auto tile_size = PrefixSumAgent<unsigned int>::tile_size;

  const auto n_tiles = (n + tile_size - 1) / tile_size;
  cu::unified_vector<unsigned int> u_auxiliary(n_tiles);

  k_PrefixSumLocal<<<n_blocks, n_threads>>>(
      u_data.data(), u_local_sums.data(), n, u_auxiliary.data());
  SYNC_DEVICE();

  for (auto tile_id = 0; tile_id < n_tiles; ++tile_id) {
    EXPECT_EQ(u_auxiliary[tile_id], tile_size);
  }

  k_SingleBlockExclusiveScan<<<1, n_threads>>>(
      u_auxiliary.data(), u_auxiliary.data(), n_tiles);
  SYNC_DEVICE();

  cu::unified_vector<unsigned int> u_global_sums(n);

  k_MakeGlobalPrefixSum<<<n_blocks, n_threads>>>(
      u_local_sums.data(), u_global_sums.data(), u_auxiliary.data(), n);
  SYNC_DEVICE();

  for (auto i = 0; i < n; ++i) {
    EXPECT_EQ(u_global_sums[i], i);
  }
}

TEST(GlobalPrefixSum, Test_GlobalPrefixSum) {
  EXPECT_NO_FATAL_FAILURE(Test_GlobalPrefixSum(1 << 10, 1));  // 1024
  EXPECT_NO_FATAL_FAILURE(Test_GlobalPrefixSum(1 << 16, 2));  // 65536
  EXPECT_NO_FATAL_FAILURE(Test_GlobalPrefixSum(1 << 20, 4));  // 1048576
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
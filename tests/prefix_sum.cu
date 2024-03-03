#include <gtest/gtest.h>

#include <cub/cub.cuh>
#include <numeric>
#include <vector>

#include "cuda/dispatchers/prefix_sum_dispatch.cuh"
#include "cuda/unified_vector.cuh"

static void Test_PrefixSum(const int n, const int n_blocks) {
  const cu::unified_vector<unsigned int> u_data(n, 1);
  // cu::unified_vector<unsigned int> u_local_sums(n);
  cu::unified_vector<unsigned int> u_output(n);

  constexpr auto tile_size = gpu::PrefixSumAgent<unsigned int>::tile_size;
  const auto n_tiles = cub::DivideAndRoundUp(n, tile_size);
  cu::unified_vector<unsigned int> u_auxiliary(n_tiles);

  cudaStream_t stream;
  CHECK_CUDA_CALL(cudaStreamCreate(&stream));

  gpu::dispatch_PrefixSum(n_blocks,
                          stream,
                          u_data.data(),
                          // u_local_sums.data(),
                          u_output.data(),
                          u_auxiliary.data(),
                          n);
  SYNC_STREAM(stream);

  for (auto i = 0; i < n; ++i) {
    EXPECT_EQ(u_output[i], i);
  }

  CHECK_CUDA_CALL(cudaStreamDestroy(stream));
}

static void Test_PrefixSumArbitaryInput_Int(const int n, const int n_blocks) {
  cu::unified_vector<int> u_data(n);
  cu::unified_vector<int> u_output(n);

  std::iota(u_data.begin(), u_data.end(), 1);
  const std::vector cpu_backup_data(u_data.begin(), u_data.end());

  std::vector<int> cpu_output(n);
  std::exclusive_scan(u_data.begin(), u_data.end(), cpu_output.begin(), 0);

  constexpr auto tile_size = gpu::PrefixSumAgent<int>::tile_size;
  const auto n_tiles = cub::DivideAndRoundUp(n, tile_size);
  cu::unified_vector<int> u_auxiliary(n_tiles);

  cudaStream_t stream;
  CHECK_CUDA_CALL(cudaStreamCreate(&stream));

  gpu::dispatch_PrefixSum(
      n_blocks, stream, u_data.data(), u_output.data(), u_auxiliary.data(), n);
  SYNC_STREAM(stream);

  auto is_equal =
      std::equal(u_output.begin(), u_output.end(), cpu_output.begin());
  EXPECT_TRUE(is_equal);

  // Also check if u_data is modified (it should not)
  is_equal = std::equal(u_data.begin(), u_data.end(), cpu_backup_data.begin());
  EXPECT_TRUE(is_equal);

  CHECK_CUDA_CALL(cudaStreamDestroy(stream));
}

TEST(GlobalPrefixSumRegular, Test_PrefixSum) {
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1 << 10, 1));  // 1024
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1 << 16, 2));  // 65536
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1 << 20, 4));  // 1048576
}

TEST(GlobalPrefixSumIrregular, Test_PrefixSum) {
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(114514, 1));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(640 * 480, 8));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1920 * 1080, 16));
}

TEST(GlobalPrefixSumArbitraryInput, Test_PrefixSum) {
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumArbitaryInput_Int(1 << 10, 1));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumArbitaryInput_Int(1 << 16, 2));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSumArbitaryInput_Int(1 << 20, 4));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
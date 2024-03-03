#include <gtest/gtest.h>

#include <cub/cub.cuh>
#include <numeric>
#include <random>
#include <vector>

#include "cuda/dispatchers/unique_dispatch.cuh"
#include "cuda/unified_vector.cuh"

static void Test_Unique(const int n, const int n_blocks) {
  cu::unified_vector<unsigned int> u_data(n);
  cu::unified_vector<unsigned int> u_output(n);

  std::mt19937 gen(114514);
  std::uniform_int_distribution<unsigned int> dis(0, n / 2);

  std::generate(
      u_data.begin(), u_data.end(), [&dis, &gen]() { return dis(gen); });
  std::sort(u_data.begin(), u_data.end());

  cudaStream_t stream;
  CHECK_CUDA_CALL(cudaStreamCreate(&stream));

  // temporary memory
  cu::unified_vector<int> u_flag_heads(n);
  const auto prefix_sum_num_tiles =
      cub::DivideAndRoundUp(n, gpu::PrefixSumAgent<int>::tile_size);
  cu::unified_vector<int> u_auxiliary(prefix_sum_num_tiles);

  gpu::dispatch_Unique(n_blocks,
                       stream,
                       u_data.data(),
                       u_output.data(),
                       u_flag_heads.data(),
                       u_auxiliary.data(),
                       n);
  SYNC_STREAM(stream);

  // check with cpu
  std::vector cpu_data(u_data.begin(), u_data.end());
  const auto it = std::unique(cpu_data.begin(), cpu_data.end());
  const auto cpu_num_unique = std::distance(cpu_data.begin(), it);

  const auto is_equal = std::equal(
      u_output.begin(), u_output.begin() + cpu_num_unique, cpu_data.begin());

  for (int i = 0; i < 2048; ++i) {
    std::cout << "[" << i << "] " << u_data[i] << " -\t" << u_output[i] << "\t"
              << cpu_data[i] << std::endl;
  }

  EXPECT_TRUE(is_equal);

  CHECK_CUDA_CALL(cudaStreamDestroy(stream));
}

TEST(Test_UniqueRegular, Test_Unique) {
  EXPECT_NO_FATAL_FAILURE(Test_Unique(1 << 10, 1));  // 1024
  EXPECT_NO_FATAL_FAILURE(Test_Unique(1 << 16, 2));  // 65536
  EXPECT_NO_FATAL_FAILURE(Test_Unique(1 << 20, 4));  // 1048576
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
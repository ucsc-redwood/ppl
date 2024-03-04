#include <gtest/gtest.h>

#include <algorithm>
#include <cub/cub.cuh>

#include "cuda/dispatchers/sort_dispatch.cuh"
#include "cuda/unified_vector.cuh"

static void Test_RadixSort(const int n, const int grid_size) {
  // Essentials
  cu::unified_vector<unsigned int> u_sort(n);

  std::generate(
      u_sort.begin(), u_sort.end(), [n = n]() mutable { return --n; });

  // Temporary storages
  constexpr auto radix = 256;
  constexpr auto passes = 4;
  const auto binning_thread_blocks = cub::DivideAndRoundUp(n, 7680);

  cu::unified_vector<unsigned int> u_sort_alt(n);
  cu::unified_vector<unsigned int> u_global_histogram(radix * passes);
  cu::unified_vector<unsigned int> u_index(passes);
  cu::unified_vector<unsigned int> u_first_pass_histogram(
      radix * binning_thread_blocks);
  cu::unified_vector<unsigned int> u_second_pass_histogram(
      radix * binning_thread_blocks);
  cu::unified_vector<unsigned int> u_third_pass_histogram(
      radix * binning_thread_blocks);
  cu::unified_vector<unsigned int> u_fourth_pass_histogram(
      radix * binning_thread_blocks);

  auto is_sorted = std::is_sorted(u_sort.begin(), u_sort.end());
  EXPECT_FALSE(is_sorted);

  cudaStream_t stream;
  CHECK_CUDA_CALL(cudaStreamCreate(&stream));

  // attach
  ATTACH_STREAM_SINGLE(u_sort.data());
  ATTACH_STREAM_SINGLE(u_sort_alt.data());
  ATTACH_STREAM_SINGLE(u_global_histogram.data());
  ATTACH_STREAM_SINGLE(u_index.data());
  ATTACH_STREAM_SINGLE(u_first_pass_histogram.data());
  ATTACH_STREAM_SINGLE(u_second_pass_histogram.data());
  ATTACH_STREAM_SINGLE(u_third_pass_histogram.data());
  ATTACH_STREAM_SINGLE(u_fourth_pass_histogram.data());

  gpu::dispatch_RadixSort(grid_size,
                          stream,
                          u_sort.data(),
                          u_sort_alt.data(),
                          u_global_histogram.data(),
                          u_index.data(),
                          u_first_pass_histogram.data(),
                          u_second_pass_histogram.data(),
                          u_third_pass_histogram.data(),
                          u_fourth_pass_histogram.data(),
                          n);

  SYNC_STREAM(stream);

  is_sorted = std::is_sorted(u_sort.begin(), u_sort.end());

  EXPECT_TRUE(is_sorted);

  CHECK_CUDA_CALL(cudaStreamDestroy(stream));
}

TEST(RadixSort_Regular, Test_RadixSort) {
  EXPECT_NO_FATAL_FAILURE(Test_RadixSort(1 << 16, 1));   // 64K
  EXPECT_NO_FATAL_FAILURE(Test_RadixSort(1 << 18, 16));  // 256K
  EXPECT_NO_FATAL_FAILURE(Test_RadixSort(1 << 20, 64));  // 1M
}

TEST(RadixSort_Irregular, Test_RadixSort) {
  EXPECT_NO_FATAL_FAILURE(Test_RadixSort(114514, 1));
  EXPECT_NO_FATAL_FAILURE(Test_RadixSort(640 * 480, 8));
  EXPECT_NO_FATAL_FAILURE(Test_RadixSort(1920 * 1080, 16));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
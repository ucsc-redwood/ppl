#include <gtest/gtest.h>

#include <cub/cub.cuh>
#include <numeric>
#include <vector>

#include "cuda/dispatchers/prefix_sum_dispatch.cuh"
#include "cuda/unified_vector.cuh"

#define CREATE_STREAM  \
  cudaStream_t stream; \
  CHECK_CUDA_CALL(cudaStreamCreate(&stream));

#define DESCROY_STREAM CHECK_CUDA_CALL(cudaStreamDestroy(stream));

// -----------------------------------------------------------------------------
// Test Local Prefix Sum
// -----------------------------------------------------------------------------

static void Test_LocalPrefixSum(const int n, const int n_blocks) {
  CREATE_STREAM;

  const cu::unified_vector<unsigned int> u_input(n, 1);
  cu::unified_vector<unsigned int> u_output(n);

  constexpr auto n_threads = gpu::PrefixSumAgent<int>::n_threads;
  constexpr auto tile_size = gpu::PrefixSumAgent<int>::tile_size;
  const auto n_tiles = cub::DivideAndRoundUp(n, tile_size);

  cu::unified_vector<unsigned int> u_auxiliary(n_tiles);

  gpu::k_PrefixSumLocal<<<n_blocks, n_threads, 0, stream>>>(
      u_input.data(), u_output.data(), n, u_auxiliary.data());
  SYNC_DEVICE();

  // Each tile should independently have a prefix sum
  for (auto i = 0; i < n_tiles; ++i) {
    const auto offset = i * tile_size;

    // locally should [0, 1, 2, ... ,tile_size - 1]
    for (auto j = 0; j < tile_size; ++j) {
      if (offset + j < n) {
        EXPECT_EQ(u_output[offset + j], j);
      }
    }
  }

  // For each tile, the auxiliary memory should be the size of a tile
  for (auto i = 0; i < n_tiles - 1; ++i) {
    EXPECT_EQ(u_auxiliary[i], tile_size);
  }
  // for last block, the number is the remainder
  const auto last_block_size = n - (n_tiles - 1) * tile_size;
  EXPECT_EQ(u_auxiliary[n_tiles - 1], last_block_size);

  DESCROY_STREAM;
}

TEST(Test_LocalPrefixSum, RegularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_LocalPrefixSum(1 << 10, 1));  // 1024
  EXPECT_NO_FATAL_FAILURE(Test_LocalPrefixSum(1 << 16, 2));  // 65536
  EXPECT_NO_FATAL_FAILURE(Test_LocalPrefixSum(1 << 20, 4));  // 1048576
}

TEST(Test_LocalPrefixSum, IrregularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_LocalPrefixSum(114514, 1));
  EXPECT_NO_FATAL_FAILURE(Test_LocalPrefixSum(640 * 480, 2));
  EXPECT_NO_FATAL_FAILURE(Test_LocalPrefixSum(1920 * 1080, 4));
}

// -----------------------------------------------------------------------------
// Single Block Exclusive Scan
// -----------------------------------------------------------------------------

static void Test_SingleBlock(const int n) {
  CREATE_STREAM;

  const cu::unified_vector<unsigned int> u_input(n, 1);
  cu::unified_vector<unsigned int> u_output(n);

  constexpr auto n_threads = gpu::PrefixSumAgent<int>::n_threads;

  gpu::k_SingleBlockExclusiveScan<<<1, n_threads, 0, stream>>>(
      u_input.data(), u_output.data(), n);
  SYNC_DEVICE();

  for (auto i = 0; i < n; ++i) {
    EXPECT_EQ(u_output[i], i);
  }

  DESCROY_STREAM;
}

static void Test_SingleBlock_Iota(const int n) {
  CREATE_STREAM;

  cu::unified_vector<unsigned int> u_input(n);
  std::iota(u_input.begin(), u_input.end(), 0);

  cu::unified_vector<unsigned int> u_output(n);
  std::vector<unsigned int> cpu_output(n);

  constexpr auto n_threads = gpu::PrefixSumAgent<int>::n_threads;

  gpu::k_SingleBlockExclusiveScan<<<1, n_threads, 0, stream>>>(
      u_input.data(), u_output.data(), n);
  SYNC_DEVICE();

  std::exclusive_scan(u_input.begin(), u_input.end(), cpu_output.begin(), 0);

  for (auto i = 0; i < n; ++i) {
    EXPECT_EQ(u_output[i], cpu_output[i]);
  }

  DESCROY_STREAM;
}

TEST(Test_SingleBlockExclusiveScan, RegularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock(1 << 10));  // 1024
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock(1 << 16));  // 65536
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock(1 << 20));  // 1048576
}

TEST(Test_SingleBlockExclusiveScan, IrregularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock(114514));
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock(640 * 480));
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock(1920 * 1080));
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock(753413));
}

TEST(Test_SingleBlockExclusiveScan, RegularInput_Arbitary) {
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock_Iota(1 << 10));  // 1024
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock_Iota(1 << 16));  // 65536
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock_Iota(1 << 20));  // 1048576
}

TEST(Test_SingleBlockExclusiveScan, IrregularInput_Arbitary) {
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock_Iota(114514));
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock_Iota(640 * 480));
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock_Iota(1920 * 1080));
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlock_Iota(753413));
}

// -----------------------------------------------------------------------------
// Global Prefix Sum
// -----------------------------------------------------------------------------

static void Test_PrefixSum(const int n, const int n_blocks) {
  CREATE_STREAM;

  const cu::unified_vector<unsigned int> u_data(n, 1);
  cu::unified_vector<unsigned int> u_output(n);

  constexpr auto tile_size = gpu::PrefixSumAgent<unsigned int>::tile_size;
  const auto n_tiles = cub::DivideAndRoundUp(n, tile_size);
  cu::unified_vector<unsigned int> u_auxiliary(n_tiles);

  gpu::dispatch_PrefixSum(
      n_blocks, stream, u_data.data(), u_output.data(), u_auxiliary.data(), n);
  SYNC_STREAM(stream);

  for (auto i = 0; i < n; ++i) {
    EXPECT_EQ(u_output[i], i);
  }

  DESCROY_STREAM;
}

static void Test_PrefixSum_Iota(const int n, const int n_blocks) {
  CREATE_STREAM;

  cu::unified_vector<unsigned int> u_data(n);
  std::iota(u_data.begin(), u_data.end(), 0);

  cu::unified_vector<unsigned int> u_output(n);
  std::vector<unsigned int> cpu_output(n);

  constexpr auto tile_size = gpu::PrefixSumAgent<unsigned int>::tile_size;
  const auto n_tiles = cub::DivideAndRoundUp(n, tile_size);
  cu::unified_vector<unsigned int> u_auxiliary(n_tiles);

  gpu::dispatch_PrefixSum(
      n_blocks, stream, u_data.data(), u_output.data(), u_auxiliary.data(), n);
  SYNC_STREAM(stream);

  std::exclusive_scan(u_data.begin(), u_data.end(), cpu_output.begin(), 0);
  // for (auto i = 0; i < n; ++i) {
  //   std::cout << i << "------ \t" << u_output[i] << " " << cpu_output[i] <<
  //   std::endl;
  // }

  for (auto i = 0; i < 2 * tile_size; ++i) {
    EXPECT_EQ(u_output[i], cpu_output[i]);
  }

  DESCROY_STREAM;
}

// static void Test_PrefixSumArbitaryInput_Int(const int n, const int n_blocks)
// {
//   CREATE_STREAM;

//   cu::unified_vector<int> u_data(n);
//   cu::unified_vector<int> u_output(n);

//   std::iota(u_data.begin(), u_data.end(), 1);
//   const std::vector cpu_backup_data(u_data.begin(), u_data.end());

//   std::vector<int> cpu_output(n);
//   std::exclusive_scan(u_data.begin(), u_data.end(), cpu_output.begin(), 0);

//   constexpr auto tile_size = gpu::PrefixSumAgent<int>::tile_size;
//   const auto n_tiles = cub::DivideAndRoundUp(n, tile_size);
//   cu::unified_vector<int> u_auxiliary(n_tiles);

//   cudaStream_t stream;
//   CHECK_CUDA_CALL(cudaStreamCreate(&stream));

//   gpu::dispatch_PrefixSum(
//       n_blocks, stream, u_data.data(), u_output.data(), u_auxiliary.data(),
//       n);
//   SYNC_STREAM(stream);

//   // auto is_equal =
//   //     std::equal(u_output.begin(), u_output.end(), cpu_output.begin());
//   // EXPECT_TRUE(is_equal);

//   for (auto i = 0; i < n; ++i) {
//     EXPECT_EQ(u_output[i], cpu_output[i]);
//   }

//   // Also check if u_data is modified (it should not)
//   auto is_equal =
//       std::equal(u_data.begin(), u_data.end(), cpu_backup_data.begin());
//   EXPECT_TRUE(is_equal);

//   DESCROY_STREAM;
// }

TEST(Test_PrefixSum, RegularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1 << 10, 1));  // 1024
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1 << 16, 2));  // 65536
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1 << 20, 4));  // 1048576
}

TEST(Test_PrefixSum, IrregularInput) {
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(114514, 1));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(640 * 480, 8));
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum(1920 * 1080, 16));
}

TEST(Test_PrefixSum, RegularInput_Arbitary) {
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum_Iota(1 << 10, 1));  // 1024
  EXPECT_NO_FATAL_FAILURE(Test_PrefixSum_Iota(1 << 16, 1));  // 65536
  // EXPECT_NO_FATAL_FAILURE(Test_PrefixSum_Iota(1 << 20, 4));  // 1048576
}

// TEST(Test_PrefixSum, Test_PrefixSumArbitaryInput_Int) {
//   EXPECT_NO_FATAL_FAILURE(Test_PrefixSumArbitaryInput_Int(1 << 10, 1));
//   EXPECT_NO_FATAL_FAILURE(Test_PrefixSumArbitaryInput_Int(1 << 16, 2));
//   EXPECT_NO_FATAL_FAILURE(Test_PrefixSumArbitaryInput_Int(1 << 20, 4));
// }

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
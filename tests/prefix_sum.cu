#include <gtest/gtest.h>

#include <cub/cub.cuh>

#include "unified_vector.cuh"

// ----------------------------------------------------------------------------
// Local Sums
// ----------------------------------------------------------------------------

template <typename T>
__global__ void k_LocalPrefixSums(const T *u_input, T *u_output, const int n) {
  constexpr auto n_threads = 128;
  constexpr auto items_per_thread = 4;
  constexpr auto tile_size = n_threads * items_per_thread;

  using BlockLoad = cub::
      BlockLoad<T, n_threads, items_per_thread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockScan = cub::BlockScan<T, n_threads>;
  using BlockStore = cub::BlockStore<T,
                                     n_threads,
                                     items_per_thread,
                                     cub::BLOCK_STORE_WARP_TRANSPOSE>;

  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockScan::TempStorage scan;
    typename BlockStore::TempStorage store;
  } temp_storage;

  const auto num_tiles = (n + tile_size - 1) / tile_size;

  for (auto tile_idx = blockIdx.x; tile_idx < num_tiles;
       tile_idx += gridDim.x) {
    T thread_data[items_per_thread];
    BlockLoad(temp_storage.load)
        .Load(u_input + tile_idx * tile_size, thread_data);
    __syncthreads();

    BlockScan(temp_storage.scan).ExclusiveSum(thread_data, thread_data);
    __syncthreads();

    BlockStore(temp_storage.store)
        .Store(u_output + tile_idx * tile_size, thread_data);
    __syncthreads();
  }
}

static void Test_PrefixSumLocal(const int n, const int n_blocks) {
  cu::unified_vector<unsigned int> u_data(n, 1);
  cu::unified_vector<unsigned int> u_output(n);

  constexpr auto n_threads = 128;

  k_LocalPrefixSums<<<n_blocks, n_threads>>>(u_data.data(), u_output.data(), n);
  SYNC_DEVICE();

  constexpr auto items_per_thread = 4;
  constexpr auto tile_size = n_threads * items_per_thread;
  const auto n_tiles = (n + tile_size - 1) / tile_size;

  for (auto tile_id = 0; tile_id < n_tiles; ++tile_id) {
    for (auto i = 0; i < tile_size; ++i) {
      const auto idx = tile_id * tile_size + i;
      if (idx < n) {
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

// ----------------------------------------------------------------------------
// Single block
// ----------------------------------------------------------------------------

template <typename T>
struct BlockPrefixCallbackOp {
  T running_total;

  __device__ BlockPrefixCallbackOp(const T running_total)
      : running_total(running_total) {}

  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide
  // scan.
  __device__ T operator()(const T block_aggregate) {
    T old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

template <typename T>
__global__ void k_SingleBlockPrefixSum(const T *u_input,
                                       T *u_output,
                                       const int n) {
  constexpr auto n_threads = 128;
  constexpr auto items_per_thread = 4;
  constexpr auto tile_size = n_threads * items_per_thread;

  using BlockLoad = cub::
      BlockLoad<T, n_threads, items_per_thread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockScan = cub::BlockScan<T, n_threads>;
  using BlockStore = cub::BlockStore<T,
                                     n_threads,
                                     items_per_thread,
                                     cub::BLOCK_STORE_WARP_TRANSPOSE>;

  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockScan::TempStorage scan;
    typename BlockStore::TempStorage store;
  } temp_storage;

  // use input[0] instead of 0
  //   BlockPrefixCallbackOp prefix_op(u_input[0]);
  BlockPrefixCallbackOp prefix_op(0);

  T thread_data[items_per_thread];

  // Have the block iterate over segments of items
  for (auto my_block_offset = 0; my_block_offset < n;
       my_block_offset += tile_size) {
    BlockLoad(temp_storage.load)
        .Load(u_input + my_block_offset, thread_data, n);
    __syncthreads();

    BlockScan(temp_storage.scan)
        .ExclusiveSum(thread_data, thread_data, prefix_op);
    __syncthreads();

    BlockStore(temp_storage.store)
        .Store(u_output + my_block_offset, thread_data, n);
    __syncthreads();
  }
}

static void Test_SingleBlockPrefixSum(const int n) {
  cu::unified_vector<unsigned int> u_data(n, 1);
  cu::unified_vector<unsigned int> u_output(n);

  constexpr auto n_threads = 128;

  k_SingleBlockPrefixSum<<<1, n_threads>>>(u_data.data(), u_output.data(), n);
  SYNC_DEVICE();

  for (auto i = 0; i < n; ++i) {
    EXPECT_EQ(u_output[i], i);
  }
}

// Test case
TEST(PrefixSumTestRegular, Test_SingleBlockPrefixSum) {
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlockPrefixSum(1 << 10));  // 1024
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlockPrefixSum(1 << 16));  // 65536
  EXPECT_NO_FATAL_FAILURE(Test_SingleBlockPrefixSum(1 << 20));  // 1048576
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

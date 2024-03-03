#include <gtest/gtest.h>

#include <cub/cub.cuh>
#include <numeric>
#include <vector>

#include "cuda/unified_vector.cuh"

// ----------------------------------------------------------------------------
// Local Sums
// ----------------------------------------------------------------------------

template <typename T>
__global__ void k_LocalPrefixSums(const T* u_input, T* u_output, const int n) {
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
  const cu::unified_vector<unsigned int> u_data(n, 1);
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

/**
 * @brief Equivalent to std::exclusive_scan(u_input, u_input + n, u_output, 0);
 * But should only use 1 single block. It runs a accumulate aggregation on the
 * blocks.
 *
 * @tparam T: The type of the input and output arrays
 * @param u_input: The input array
 * @param u_output: The output array
 * @param n: The size of the input and output arrays
 * @param init: The initial value for the prefix sum, defaults to 0
 * @return
 */
template <typename T>
__global__ void k_SingleBlockExclusiveScan(const T* u_input,
                                           T* u_output,
                                           const int n,
                                           const T init = T()) {
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
  BlockPrefixCallbackOp prefix_op(init);

  // Have the block iterate over segments of items
  for (auto my_block_offset = 0; my_block_offset < n;
       my_block_offset += tile_size) {
    T thread_data[items_per_thread];

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

  constexpr auto n_threads = 128;

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

// ----------------------------------------------------------------------------
// Global Prefix Sum
// ----------------------------------------------------------------------------

template <typename T>
__global__ void k_LocalPrefixSums_AndSaveLastElement(const T* u_input,
                                                     T* u_output,
                                                     T* u_auxiliary,
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

  const auto num_tiles = (n + tile_size - 1) / tile_size;

  for (auto tile_idx = blockIdx.x; tile_idx < num_tiles;
       tile_idx += gridDim.x) {
    T thread_data[items_per_thread];
    BlockLoad(temp_storage.load)
        .Load(u_input + tile_idx * tile_size, thread_data);
    __syncthreads();

    T aggregate;
    BlockScan(temp_storage.scan)
        .ExclusiveSum(thread_data, thread_data, aggregate);
    __syncthreads();

    // save aggregate to u_auxiliary[tile_idx] if u_auxiliary is not nullptr
    if (u_auxiliary && threadIdx.x == 0) {
      u_auxiliary[tile_idx] = aggregate;
    }

    BlockStore(temp_storage.store)
        .Store(u_output + tile_idx * tile_size, thread_data);
    __syncthreads();
  }

  __syncthreads();
}

template <typename T>
__global__ void k_MakeGlobalPrefixSum(T* u_local_sums,
                                      T* u_global_sums,
                                      T* u_auxiliary_summed,
                                      const int n) {
  constexpr auto n_threads = 128;
  constexpr auto items_per_thread = 4;
  constexpr auto tile_size = n_threads * items_per_thread;

  using BlockLoad = cub::
      BlockLoad<T, n_threads, items_per_thread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockStore = cub::BlockStore<T,
                                     n_threads,
                                     items_per_thread,
                                     cub::BLOCK_STORE_WARP_TRANSPOSE>;

  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
  } temp_storage;

  const auto num_tiles = (n + tile_size - 1) / tile_size;

  for (auto tile_idx = blockIdx.x; tile_idx < num_tiles;
       tile_idx += gridDim.x) {
    T thread_data[items_per_thread];
    BlockLoad(temp_storage.load)
        .Load(u_local_sums + tile_idx * tile_size, thread_data);
    __syncthreads();

#pragma unroll
    for (auto i = 0; i < items_per_thread; ++i) {
      const auto idx = tile_idx * tile_size + i;
      if (idx < n) {
        thread_data[i] += u_auxiliary_summed[tile_idx];
      }
    }

    BlockStore(temp_storage.store)
        .Store(u_global_sums + tile_idx * tile_size, thread_data, n);
    __syncthreads();
  }

  //   const auto index = blockIdx.x * blockDim.x + threadIdx.x;

  //   if (index < n) {
  //     const int offset = last_elements_sumed[blockIdx.x];
  //     global_prefix_sum[index] = local_prefix_sums[index] + offset;
  //   }
}

static void Test_GlobalPrefixSum(const int n, const int n_blocks) {
  const cu::unified_vector<unsigned int> u_data(n, 1);
  cu::unified_vector<unsigned int> u_local_sums(n);

  constexpr auto n_threads = 128;
  constexpr auto items_per_thread = 4;
  constexpr auto tile_size = n_threads * items_per_thread;

  const auto n_tiles = (n + tile_size - 1) / tile_size;
  cu::unified_vector<unsigned int> u_auxiliary(n_tiles);

  k_LocalPrefixSums_AndSaveLastElement<<<n_blocks, n_threads>>>(
      u_data.data(), u_local_sums.data(), u_auxiliary.data(), n);
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
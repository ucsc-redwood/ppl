#include <cub/cub.cuh>

namespace gpu {

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

  for (auto my_block_idx = blockIdx.x; my_block_idx < num_tiles;
       my_block_idx += gridDim.x) {
    T thread_data[items_per_thread];
    BlockLoad(temp_storage.load)
        .Load(u_input + my_block_idx * tile_size, thread_data);
    __syncthreads();

    BlockScan(temp_storage.scan).ExclusiveSum(thread_data, thread_data);
    __syncthreads();

    BlockStore(temp_storage.store)
        .Store(u_output + my_block_idx * tile_size, thread_data);
    __syncthreads();
  }
}

// =============================================================================
// Explicit Instantiation (int, unsigned int)
// =============================================================================

// for find duplicate kernel
template __global__ void k_LocalPrefixSums(const unsigned int *u_input,
                                           unsigned int *u_output,
                                           const int n);

// for edge count kernel
template __global__ void k_LocalPrefixSums(const int *u_input,
                                           int *u_output,
                                           const int n);

}  // namespace gpu
#include <cub/cub.cuh>

namespace gpu {

template <typename T>
__global__ void k_BetterFindDups(const T *u_keys,
                                 int *u_flag_heads,
                                 const int n) {
  constexpr auto n_threads = 256;
  constexpr auto items_per_thread = 4;
  constexpr auto tile_size = n_threads * items_per_thread;

  using BlockLoad = cub::
      BlockLoad<T, n_threads, items_per_thread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockDiscontinuity = cub::BlockDiscontinuity<T, n_threads>;
  using BlockStore = cub::BlockStore<int,
                                     n_threads,
                                     items_per_thread,
                                     cub::BLOCK_STORE_WARP_TRANSPOSE>;

  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockDiscontinuity::TempStorage discontinuity;
    typename BlockStore::TempStorage store;
  } temp_storage;

  const auto num_tiles = (n + tile_size - 1) / tile_size;
  for (auto my_block_id = blockIdx.x; my_block_id < num_tiles;
       my_block_id += gridDim.x) {
    const auto tile_offset = my_block_id * tile_size;

    if (threadIdx.x == 0)
      printf("[block %d] is processing block (%d/%d)\n",
             blockIdx.x,
             my_block_id,
             num_tiles);

    // Have thread0 obtain the predecessor item for the entire tile
    int tile_predecessor_item;
    if (threadIdx.x == 0)
      tile_predecessor_item =
          (tile_offset == 0) ? u_keys[0] : u_keys[tile_offset - 1];

    T thread_data[items_per_thread];
    int head_flags[items_per_thread];

    BlockLoad(temp_storage.load).Load(u_keys + tile_offset, thread_data, n);
    __syncthreads();

    BlockDiscontinuity(temp_storage.discontinuity)
        .FlagHeads(
            head_flags, thread_data, cub::Inequality(), tile_predecessor_item);
    __syncthreads();

    BlockStore(temp_storage.store)
        .Store(u_flag_heads + tile_offset, head_flags, n);
    __syncthreads();
  }
}

template <typename T>
__global__ void k_MoveDups(const T *u_keys,
                           const int *u_flag_heads_sums,
                           const int N /*old n*/,
                           T *u_keys_out,
                           int *n_unique_out) {
  const auto num_tiles = (N + blockDim.x - 1) / blockDim.x;

  for (auto my_block_id = blockIdx.x; my_block_id < num_tiles;
       my_block_id += gridDim.x) {
    const auto idx = my_block_id * blockDim.x + threadIdx.x;

    u_keys_out[u_flag_heads_sums[idx]] = u_keys[idx];
  }

  // we want to save number of unique elements
  if (threadIdx.x == 0) {
    *n_unique_out = u_flag_heads_sums[N - 1] + 1;
  }
}

template __global__ void k_BetterFindDups(const unsigned int *u_keys,
                                          int *u_flag_heads,
                                          const int n);

}  // namespace gpu
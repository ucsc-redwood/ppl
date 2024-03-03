#pragma once

#include <spdlog/spdlog.h>

#include "cuda/agents/prefix_sum_agent.cuh"
#include "cuda/kernels/06_prefix_sum.cuh"

namespace gpu {

/** @brief Dispatches the prefix sum computation. However, you need to allocate
 * a temporary memory for the auxiliary data. The size of the auxiliary memory
 * should be the size of a tile (# of threads X # of items per thread).
 *
 * @tparam T The type of the data.
 * @param grid_size The number of blocks you want to use in the grid.
 * @param stream The CUDA stream.
 * @param u_data The input data.
 * @param u_global_sums The global prefix sum.
 * @param u_auxiliary The auxiliary memory. Need to be the size of a tile (# of
 * threads X # of items per thread).
 * @param n The size of the input data.
 */
template <typename T>
void dispatch_PrefixSum(const int grid_size,
                        const cudaStream_t stream,
                        const T* u_data,
                        T* u_global_sums,
                        T* u_auxiliary,
                        const int n) {
  constexpr auto n_threads = PrefixSumAgent<T>::n_threads;
  constexpr auto tile_size = PrefixSumAgent<T>::tile_size;

  const auto n_tiles = cub::DivideAndRoundUp(n, tile_size);

  spdlog::debug("Dispatching k_PrefixSumLocal with ({} blocks, {} threads)",
                grid_size,
                n_threads);

  k_PrefixSumLocal<<<grid_size, n_threads, 0, stream>>>(
      u_data, u_global_sums, n, u_auxiliary);

  spdlog::debug(
      "Dispatching k_SingleBlockExclusiveScan with (1 blocks, {} "
      "threads)",
      n_threads);

  k_SingleBlockExclusiveScan<<<1, n_threads, 0, stream>>>(
      u_auxiliary, u_auxiliary, n_tiles);

  spdlog::debug(
      "Dispatching k_MakeGlobalPrefixSum with ({} blocks, {} "
      "threads)",
      grid_size,
      n_threads);

  k_MakeGlobalPrefixSum<<<grid_size, n_threads, 0, stream>>>(
      u_global_sums, u_auxiliary, u_global_sums, n);
}

template <typename T>
[[deprecated(
    "Use dispatch_PrefixSum with the const u_data instead. This one uses "
    "necessary extra temporary memory")]] void
dispatch_PrefixSum(const int grid_size,
                   const cudaStream_t stream,
                   const T* u_data,
                   T* u_local_sums,
                   T* u_global_sums,
                   T* u_auxiliary,
                   const int n) {
  constexpr auto n_threads = PrefixSumAgent<T>::n_threads;
  constexpr auto tile_size = PrefixSumAgent<T>::tile_size;

  const auto n_tiles = cub::DivideAndRoundUp(n, tile_size);

  k_PrefixSumLocal<<<grid_size, n_threads, 0, stream>>>(
      u_data, u_local_sums, n, u_auxiliary);

  k_SingleBlockExclusiveScan<<<1, n_threads, 0, stream>>>(
      u_auxiliary, u_auxiliary, n_tiles);

  k_MakeGlobalPrefixSum<<<grid_size, n_threads, 0, stream>>>(
      u_local_sums, u_auxiliary, u_global_sums, n);
}

}  // namespace gpu
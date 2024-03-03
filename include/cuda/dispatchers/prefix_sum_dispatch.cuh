#pragma once

#include "cuda/agents/prefix_sum_agent.cuh"
#include "cuda/kernels/prefix_sum.cuh"

namespace gpu {

//// Note: This function modifies the input array 'u_data' to local prefix sum
// template <typename T>
//[[deprecated("Use dispatch_PrefixSum with the const u_data instead")]]
// void dispatch_PrefixSum(const int grid_size,
//                         const cudaStream_t stream,
//                         T* u_data,
//                         T* u_global_sums,
//                         T* u_auxiliary,
//                         const int n) {
//   constexpr auto n_threads = PrefixSumAgent<T>::n_threads;
//   constexpr auto tile_size = PrefixSumAgent<T>::tile_size;
//
//   const auto n_tiles = cub::DivideAndRoundUp(n, tile_size);
//
//   k_PrefixSumLocal<<<grid_size, n_threads, 0, stream>>>(
//       u_data, u_data, n, u_auxiliary);
//
//   k_SingleBlockExclusiveScan<<<1, n_threads, 0, stream>>>(
//       u_auxiliary, u_auxiliary, n_tiles);
//
//   k_MakeGlobalPrefixSum<<<grid_size, n_threads, 0, stream>>>(
//       u_data, u_auxiliary, u_global_sums, n);
// }

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

  k_PrefixSumLocal<<<grid_size, n_threads, 0, stream>>>(
      u_data, u_global_sums, n, u_auxiliary);

  k_SingleBlockExclusiveScan<<<1, n_threads, 0, stream>>>(
      u_auxiliary, u_auxiliary, n_tiles);

  k_MakeGlobalPrefixSum<<<grid_size, n_threads, 0, stream>>>(
      u_global_sums, u_auxiliary, u_global_sums, n);
}

template <typename T>
[[deprecated("Use dispatch_PrefixSum with the const u_data instead")]] void
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
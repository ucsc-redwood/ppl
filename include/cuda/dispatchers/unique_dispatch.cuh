#pragma once

#include <spdlog/spdlog.h>

#include <numeric>

#include "cuda/agents/unique_agent.cuh"
#include "cuda/helper.cuh"
#include "cuda/kernels/03_unique.cuh"
#include "prefix_sum_dispatch.cuh"

namespace gpu {

inline void dispatch_Unique_easy(const int grid_size,
                                 const cudaStream_t stream,
                                 const unsigned int* u_keys,
                                 unsigned int* u_keys_out,
                                 int* u_flag_heads,
                                 const int n) {
  constexpr auto n_threads = UniqueAgent::n_threads;

  spdlog::debug("Dispatching k_FindDups with ({} blocks, {} threads)",
                grid_size,
                n_threads);

  k_FindDups<<<grid_size, n_threads, 0, stream>>>(u_keys, u_flag_heads, n);

  // In easy version, we use cpu partial_sum instead of gpu prefix sum
  SYNC_STREAM(stream);
  std::partial_sum(u_flag_heads, u_flag_heads + n, u_flag_heads);

  spdlog::debug("Dispatching k_MoveDups with ({} blocks, {} threads)",
                grid_size,
                n_threads);

  k_MoveDups<<<grid_size, n_threads, 0, stream>>>(
      u_keys, u_flag_heads, n, u_keys_out, nullptr);
}

/**
 * @brief Remove duplicate elements from a sorted array. Need to supply a
 * 'flag_heads' array of size 'n' ints for temporary storage.
 *
 * @param grid_size
 * @param stream
 * @param u_keys
 * @param u_keys_out
 * @param u_flag_heads
 * @param n
 * @param num_unique_out
 */
inline void dispatch_Unique_easy(const int grid_size,
                                 const cudaStream_t stream,
                                 const unsigned int* u_keys,
                                 unsigned int* u_keys_out,
                                 int* u_flag_heads,
                                 const int n,
                                 int& num_unique_out) {
  dispatch_Unique_easy(grid_size, stream, u_keys, u_keys_out, u_flag_heads, n);
  SYNC_STREAM(stream);

  num_unique_out = u_flag_heads[n - 1] + 1;
}

[[deprecated("currently broken")]] inline void dispatch_Unique(
    const int grid_size,
    const cudaStream_t stream,
    const unsigned int* u_keys,
    unsigned int* u_keys_out,
    int* u_flag_heads,
    int* u_auxiliary,
    const int n) {
  constexpr auto n_threads = UniqueAgent::n_threads;

  k_FindDups<<<grid_size, n_threads, 0, stream>>>(u_keys, u_flag_heads, n);

  dispatch_PrefixSum(
      grid_size, stream, u_flag_heads, u_flag_heads, u_auxiliary, n);

  k_MoveDups<<<grid_size, n_threads, 0, stream>>>(
      u_keys, u_flag_heads, n, u_keys_out, nullptr);
}
}  // namespace gpu
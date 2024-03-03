#pragma once

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <numeric>
#include <utility>

// #include "gpu_kernels/all.cuh"
#include "cuda/kernels/00_init.cuh"
#include "cuda/kernels/01_morton.cuh"
#include "cuda/kernels/02_sort.cuh"
#include "cuda/kernels/03_unique.cuh"
#include "types/one_sweep.cuh"
#include "types/unique.cuh"

namespace gpu {

// For simple kernels that don't require any special configuration
// e.g. k_InitRandomVec4,
template <typename Func, typename... Args>
void dispatchKernel(Func&& func,
                    const int grid_size,
                    const cudaStream_t stream,
                    Args&&... args) {
  // todo: lookup from table
  constexpr auto block_size = 768;

  spdlog::debug(
      "Dispatching kernel with ({} blocks, {} threads)", grid_size, block_size);

  std::forward<Func>(func)<<<grid_size, block_size, 0, stream>>>(
      std::forward<Args>(args)...);
}

// =============================================================================
// Radix Sort Kernel Dispatch
// =============================================================================

inline void dispatch_RadixSort(const int grid_size,
                               const cudaStream_t stream,
                               OneSweep& one_sweep) {
  constexpr int globalHistThreads = 128;
  constexpr int binningThreads = 512;

  const auto n = one_sweep.u_sort.size();

  spdlog::debug(
      "Dispatching k_GlobalHistogram with ({} "
      "blocks, {} threads, locial blocks: {})",
      grid_size,
      globalHistThreads,
      OneSweep::globalHistThreadblocks(n));

  k_GlobalHistogram<<<OneSweep::globalHistThreadblocks(n),
                      globalHistThreads,
                      0,
                      stream>>>(
      one_sweep.u_sort.data(), one_sweep.u_global_histogram.data(), n);

  spdlog::debug(
      "Dispatching k_DigitBinning with ({} "
      "blocks, {} threads, logical blocks: {})",
      grid_size,
      binningThreads,
      OneSweep::binningThreadblocks(n));

  k_DigitBinning<<<OneSweep::binningThreadblocks(n),
                   binningThreads,
                   0,
                   stream>>>(one_sweep.u_global_histogram.data(),
                             one_sweep.u_sort.data(),
                             one_sweep.u_sort_alt.data(),
                             one_sweep.u_pass_histogram[0].data(),
                             one_sweep.u_index.data(),
                             n,
                             0);

  k_DigitBinning<<<OneSweep::binningThreadblocks(n),
                   binningThreads,
                   0,
                   stream>>>(one_sweep.u_global_histogram.data(),
                             one_sweep.u_sort_alt.data(),
                             one_sweep.u_sort.data(),
                             one_sweep.u_pass_histogram[1].data(),
                             one_sweep.u_index.data(),
                             n,
                             8);

  k_DigitBinning<<<OneSweep::binningThreadblocks(n),
                   binningThreads,
                   0,
                   stream>>>(one_sweep.u_global_histogram.data(),
                             one_sweep.u_sort.data(),
                             one_sweep.u_sort_alt.data(),
                             one_sweep.u_pass_histogram[2].data(),
                             one_sweep.u_index.data(),
                             n,
                             16);

  k_DigitBinning<<<OneSweep::binningThreadblocks(n),
                   binningThreads,
                   0,
                   stream>>>(one_sweep.u_global_histogram.data(),
                             one_sweep.u_sort_alt.data(),
                             one_sweep.u_sort.data(),
                             one_sweep.u_pass_histogram[3].data(),
                             one_sweep.u_index.data(),
                             n,
                             24);
}

}  // namespace gpu

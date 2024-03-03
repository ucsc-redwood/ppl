#pragma once

#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>

#include "cuda/kernels/02_sort.cuh"

namespace gpu {

// Note this will modify the input data, performing an in-place sort
void dispatch_RadixSort(const int grid_size,
                        const cudaStream_t stream,
                        unsigned int* u_sort,
                        unsigned int* u_sort_alt,
                        unsigned int* u_global_histogram,
                        unsigned int* u_index,
                        unsigned int* u_first_pass_histogram,
                        unsigned int* u_second_pass_histogram,
                        unsigned int* u_third_pass_histogram,
                        unsigned int* u_fourth_pass_histogram,
                        const int n) {
  // need to match the .cu file
  constexpr int global_hist_threads = 128;
  constexpr int binning_threads = 512;

  spdlog::debug(
      "Dispatching k_GlobalHistogram with ({} "
      "blocks, {} threads)",
      grid_size,
      global_hist_threads);

  k_GlobalHistogram<<<grid_size, global_hist_threads, 0, stream>>>(
      u_sort, u_global_histogram, n);

  spdlog::debug(
      "Dispatching k_Scan with ({} "
      "blocks, {} threads)",
      4,
      256);

  k_Scan<<<4, 256, 0, stream>>>(u_global_histogram,
                                u_first_pass_histogram,
                                u_second_pass_histogram,
                                u_third_pass_histogram,
                                u_fourth_pass_histogram);
  spdlog::debug(
      "Dispatching k_DigitBinningPass with ({} "
      "blocks, {} threads) x 4...",
      grid_size,
      binning_threads);

  k_DigitBinningPass<<<grid_size, binning_threads, 0, stream>>>(
      u_sort, u_sort_alt, u_first_pass_histogram, u_index, n, 0);

  k_DigitBinningPass<<<grid_size, binning_threads, 0, stream>>>(
      u_sort_alt, u_sort, u_second_pass_histogram, u_index, n, 8);

  k_DigitBinningPass<<<grid_size, binning_threads, 0, stream>>>(
      u_sort, u_sort_alt, u_third_pass_histogram, u_index, n, 16);

  k_DigitBinningPass<<<grid_size, binning_threads, 0, stream>>>(
      u_sort_alt, u_sort, u_fourth_pass_histogram, u_index, n, 24);
}

}  // namespace gpu
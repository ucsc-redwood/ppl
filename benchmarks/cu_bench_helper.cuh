#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include "cuda/helper.cuh"
#include "kernels_fwd.h"

namespace bm = benchmark;

template <class T>
[[nodiscard]] int determineBlockSize(T func) {
  int block_size = 1;
  int min_grid_size = 1;
  CHECK_CUDA_CALL(
      cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, func));
  return block_size;
}

template <class T>
[[nodiscard]] int determineBlockSizeAndDisplay(T func, bm::State &state) {
  const auto block_size = determineBlockSize(func);
  state.counters["block_size"] = block_size;
  return block_size;
}

#define CREATE_STREAM  \
  cudaStream_t stream; \
  CHECK_CUDA_CALL(cudaStreamCreate(&stream));

#define DESCROY_STREAM CHECK_CUDA_CALL(cudaStreamDestroy(stream));

class CudaEventTimer {
 public:
  explicit CudaEventTimer(bm::State &state,
                          const bool flush_l2_cache = false,
                          const cudaStream_t stream = nullptr)
      : stream_(stream), p_state_(&state) {
    // flush all of L2$
    if (flush_l2_cache) {
      int current_device = 0;
      CHECK_CUDA_CALL(cudaGetDevice(&current_device));

      int l2_cache_bytes = 0;
      CHECK_CUDA_CALL(cudaDeviceGetAttribute(
          &l2_cache_bytes, cudaDevAttrL2CacheSize, current_device));

      if (l2_cache_bytes > 0) {
        const int memset_value = 0;
        int *l2_cache_buffer = nullptr;
        CHECK_CUDA_CALL(cudaMalloc(reinterpret_cast<void **>(&l2_cache_buffer),
                                   l2_cache_bytes));
        CHECK_CUDA_CALL(cudaMemsetAsync(
            l2_cache_buffer, memset_value, l2_cache_bytes, stream_));
        CHECK_CUDA_CALL(cudaFree(l2_cache_buffer));
      }
    }

    CHECK_CUDA_CALL(cudaEventCreate(&start_));
    CHECK_CUDA_CALL(cudaEventCreate(&stop_));
    CHECK_CUDA_CALL(cudaEventRecord(start_, stream_));
  }

  CudaEventTimer() = delete;

  /**
   * @brief Destroy the `CudaEventTimer` and ending the manual time range.
   *
   */
  ~CudaEventTimer() {
    CHECK_CUDA_CALL(cudaEventRecord(stop_, stream_));
    CHECK_CUDA_CALL(cudaEventSynchronize(stop_));
    float milliseconds = 0.0f;
    CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start_, stop_));
    p_state_->SetIterationTime(milliseconds / 1000.0f);
    CHECK_CUDA_CALL(cudaEventDestroy(start_));
    CHECK_CUDA_CALL(cudaEventDestroy(stop_));
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  cudaStream_t stream_;
  bm::State *p_state_;
};
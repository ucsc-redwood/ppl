#pragma once

// if in release mode, we don't want to check for cuda errors
#ifdef NDEBUG
#define CHECK_CUDA_CALL(x) x
#else
#include "cuda/common/helper_cuda.hpp"
#define CHECK_CUDA_CALL(x) checkCudaErrors(x)
#endif

template <typename T>
constexpr void MallocManaged(T **ptr, const size_t num_items) {
  CHECK_CUDA_CALL(
      cudaMallocManaged(reinterpret_cast<void **>(ptr), num_items * sizeof(T)));
}

template <typename T>
constexpr void MallocDevice(T **ptr, const size_t num_items) {
  CHECK_CUDA_CALL(
      cudaMalloc(reinterpret_cast<void **>(ptr), num_items * sizeof(T)));
}

template <>
inline void MallocDevice<void>(void **ptr, const size_t num_items) {
  CHECK_CUDA_CALL(cudaMalloc(ptr, num_items));
}

#define CUDA_FREE(ptr) CHECK_CUDA_CALL(cudaFree(ptr))

#define ATTACH_STREAM_SINGLE(ptr) \
  CHECK_CUDA_CALL(cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachSingle))

#define SYNC_STREAM(stream) CHECK_CUDA_CALL(cudaStreamSynchronize(stream))

#define SYNC_DEVICE() CHECK_CUDA_CALL(cudaDeviceSynchronize())

template <typename T, typename Alloc>
[[nodiscard]] std::size_t calculateMemorySize(
    const std::vector<T, Alloc> &vec) {
  return sizeof(T) * vec.size();
}

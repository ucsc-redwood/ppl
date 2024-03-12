#pragma once

#include <cuda_runtime_api.h>

// if in release mode, we don't want to check for cuda errors
#ifdef NDEBUG
#define CHECK_CUDA_CALL(x) x
#else
#include "cuda/common/helper_cuda.hpp"
#define CHECK_CUDA_CALL(x) checkCudaErrors(x)
#endif

template <typename T>
constexpr void mallocManaged(T **ptr, const size_t num_items) {
  CHECK_CUDA_CALL(cudaMallocManaged(reinterpret_cast<void **>(ptr),
                                    num_items * sizeof(T),
                                    cudaMemAttachHost));
}

template <typename T>
constexpr void mallocDevice(T **ptr, const size_t num_items) {
  CHECK_CUDA_CALL(
      cudaMalloc(reinterpret_cast<void **>(ptr), num_items * sizeof(T)));
}

#define MALLOC_MANAGED(ptr, num_items) mallocManaged(ptr, num_items)

#define MALLOC_DEVICE(ptr, num_items) mallocDevice(ptr, num_items)

#define CUDA_FREE(ptr) CHECK_CUDA_CALL(cudaFree(ptr))

#define ATTACH_STREAM_SINGLE(ptr) \
  CHECK_CUDA_CALL(cudaStreamAttachMemAsync(stream, ptr, 0))

#define ATTACH_STREAM_GLOBAL(ptr) \
  CHECK_CUDA_CALL(cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachGlobal))

#define ATTACH_STREAM_HOST(ptr) \
  CHECK_CUDA_CALL(cudaStreamAttachMemAsync(stream, ptr, 0, cudaMemAttachHost))

#define SYNC_STREAM(stream) CHECK_CUDA_CALL(cudaStreamSynchronize(stream))

#define SYNC_DEVICE() CHECK_CUDA_CALL(cudaDeviceSynchronize())

namespace cu {

template <typename T>
[[nodiscard]] size_t calcMem([[maybe_unused]] T *t, const size_t n) {
  return sizeof(T) * n;
}

template <typename T>
void clearMem(T *t, const size_t n) {
  CHECK_CUDA_CALL(cudaMemset(t, 0, calcMem(t, n)));
}

}  // namespace cu

// #define CALC_MEM(ptr, n) (sizeof(std::remove_pointer_t<decltype(ptr)>) * n)
//
#define SET_MEM_2_ZERO(ptr, item_count) \
  CHECK_CUDA_CALL(cudaMemsetAsync(      \
      ptr, 0, sizeof(std::remove_pointer_t<decltype(ptr)>) * item_count))

inline void warmUpGPU() {
  // emptyKernel<<<1, 1>>>();
  SYNC_DEVICE();
}

inline void printDeviceProperties() {
  int deviceCount;
  CHECK_CUDA_CALL(cudaGetDeviceCount(&deviceCount));
  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d has compute capability %d.%d.\n",
           dev,
           deviceProp.major,
           deviceProp.minor);
    printf("Concurrent Managed Access: %d\n",
           deviceProp.concurrentManagedAccess);
  }
}
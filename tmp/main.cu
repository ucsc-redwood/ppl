#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "cuda/helper.cuh"

__global__ void test_kernel(float* u_input, const int n) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    u_input[idx] = 1.0f;
  }
}

int main() {
  const int n = 1000;

  cudaStream_t stream;
  CHECK_CUDA_CALL(cudaStreamCreate(&stream));

  float* u_input;
  CHECK_CUDA_CALL(cudaMallocManaged(&u_input, n * sizeof(float)));

  CHECK_CUDA_CALL(
      cudaStreamAttachMemAsync(stream, u_input, 0, cudaMemAttachSingle));

  test_kernel<<<1, 256, 0, stream>>>(u_input, n);
  SYNC_STREAM(stream);

  // peek 10 elements
  for (int i = 0; i < 10; i++) {
    std::cout << u_input[i] << std::endl;
  }

  CHECK_CUDA_CALL(cudaFree(u_input));
  CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  return 0;
}
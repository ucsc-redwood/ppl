#include <cub/cub.cuh>
#include <iostream>

#include "cuda/dispatchers/prefix_sum_dispatch.cuh"
#include "cuda/unified_vector.cuh"

int main() {
  constexpr auto n = 1 << 16;  // 65536
  constexpr auto n_blocks = 16;

  cu::unified_vector<unsigned int> u_data(n, 1);
  cu::unified_vector<unsigned int> u_output(n);

  constexpr auto tile_size = gpu::PrefixSumAgent<unsigned int>::tile_size;
  const auto n_tiles = cub::DivideAndRoundUp(n, tile_size);
  cu::unified_vector<unsigned int> u_auxiliary(n_tiles);

  cudaStream_t stream;
  CHECK_CUDA_CALL(cudaStreamCreate(&stream));

  gpu::dispatch_PrefixSum(
      n_blocks, stream, u_data.data(), u_output.data(), u_auxiliary.data(), n);
  SYNC_STREAM(stream);

  // peek 32 elements
  std::copy(u_output.begin(),
            u_output.begin() + 2048,
            std::ostream_iterator<unsigned int>(std::cout, " "));

  CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  return 0;
}
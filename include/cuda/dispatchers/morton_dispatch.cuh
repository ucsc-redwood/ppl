#pragma once

#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>

#include "cuda/kernels/01_morton.cuh"

namespace gpu {

inline void dispatch_ComputeMorton(const int grid_size,
                                   const cudaStream_t stream,
                                   const glm::vec4* data,
                                   unsigned int* morton_keys,
                                   size_t n,
                                   const float min_coord,
                                   const float range) {
  constexpr auto block_size = 768;

  spdlog::debug(
      "Dispatching k_ComputeMortonCode with ({} blocks, {} "
      "threads) "
      "on {} items",
      grid_size,
      block_size,
      n);

  k_ComputeMortonCode<<<grid_size, block_size, 0, stream>>>(
      data, morton_keys, n, min_coord, range);
}

}  // namespace gpu
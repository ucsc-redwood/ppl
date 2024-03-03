#pragma once

#include <spdlog/spdlog.h>

#include "cuda/kernels/00_init.cuh"

namespace gpu {

void dispatch_InitRandomVec4(const int grid_size,
                             const cudaStream_t stream,
                             glm::vec4* u_points,
                             int n,
                             float min,
                             float range,
                             int seed) {
  constexpr auto block_size = 512;

  spdlog::debug(
      "[stream {}] Dispatching k_InitRandomVec4 with ({} blocks, {} threads) "
      "on {} items",
      (int)stream,
      grid_size,
      block_size,
      n);

  k_InitRandomVec4<<<grid_size, block_size, 0, stream>>>(
      u_points, n, min, range, seed);
}

}  // namespace gpu
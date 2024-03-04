#pragma once

#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>

#include "cuda/kernels/05_edge_count.cuh"

namespace gpu {

inline void dispatch_EdgeCount(const int grid_size,
                               const cudaStream_t stream,
                               const uint8_t* prefix_n,
                               const int* parents,
                               int* edge_count,
                               int n_brt_nodes) {
  constexpr auto block_size = 512;

  spdlog::debug(
      "Dispatching k_EdgeCount with ({} blocks, {} threads) "
      "on {} items",
      grid_size,
      block_size,
      n_brt_nodes);

  k_EdgeCount<<<grid_size, block_size, 0, stream>>>(
      prefix_n, parents, edge_count, n_brt_nodes);
}

}  // namespace gpu
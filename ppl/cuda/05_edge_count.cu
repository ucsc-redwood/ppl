#include <device_launch_parameters.h>

#include "shared/edge_count.h"

namespace gpu {

__global__ void k_EdgeCount(const uint8_t* prefix_n,
                            const int* parents,
                            int* edge_count,
                            const int n_brt_nodes) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  for (auto i = idx; i < n_brt_nodes; i += stride) {
    shared::processEdgeCount(i, prefix_n, parents, edge_count);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    edge_count[0] = 0;
  }
}

}  // namespace gpu

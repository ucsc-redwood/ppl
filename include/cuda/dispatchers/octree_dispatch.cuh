#pragma once

#include <cuda_runtime_api.h>
#include <spdlog/spdlog.h>

#include "cuda/kernels/07_octree.cuh"

namespace gpu {
inline void dispatch_BuildOctree(
    const int grid_size,
    const cudaStream_t stream,
    // --- output parameters
    int (*u_children)[8],
    glm::vec4* u_corner,
    float* u_cell_size,
    int* u_child_node_mask,
    int* u_child_leaf_mask,
    // --- end output parameters, begin input parameters (read-only)
    const int* node_offsets,
    const int* node_counts,
    const unsigned int* codes,
    const uint8_t* rt_prefixN,
    const bool* rt_hasLeafLeft,
    const bool* rt_hasLeafRight,
    const int* rt_parents,
    const int* rt_leftChild,
    const float min_coord,
    const float range,
    const int n_brt_nodes) {
  constexpr auto block_size = 512;

  spdlog::debug(
      "Dispatching k_MakeOctNodes with ({} blocks, {} threads) "
      "on {} items",
      grid_size,
      block_size,
      n_brt_nodes);

  k_MakeOctNodes<<<grid_size, block_size, 0, stream>>>(u_children,
                                                       u_corner,
                                                       u_cell_size,
                                                       u_child_node_mask,
                                                       node_offsets,
                                                       node_counts,
                                                       codes,
                                                       rt_prefixN,
                                                       rt_parents,
                                                       min_coord,
                                                       range,
                                                       n_brt_nodes);

  // spdlog::debug(
  //     "Dispatching k_LinkLeafNodes with ({} blocks, {} threads) "
  //     "on {} items",
  //     grid_size,
  //     block_size,
  //     n_brt_nodes);

  // k_LinkLeafNodes<<<grid_size, block_size, 0, stream>>>(u_children,
  //                                                       u_child_leaf_mask,
  //                                                       node_offsets,
  //                                                       node_counts,
  //                                                       codes,
  //                                                       rt_hasLeafLeft,
  //                                                       rt_hasLeafRight,
  //                                                       rt_prefixN,
  //                                                       rt_parents,
  //                                                       rt_leftChild,
  //                                                       n_brt_nodes);
}
}  // namespace gpu
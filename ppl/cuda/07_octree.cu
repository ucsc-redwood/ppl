#include <device_launch_parameters.h>

#include "cuda/kernels/07_octree.cuh"
#include "shared/oct_v2.h"

__global__ void gpu::k_MakeOctNodes(int (*u_children)[8],
                                    glm::vec4* u_corner,
                                    float* u_cell_size,
                                    int* u_child_node_mask,
                                    const int* node_offsets,  // prefix sum
                                    const int* node_counts,   // edge count
                                    const unsigned int* codes,
                                    const uint8_t* rt_prefix_n,
                                    const int* rt_parents,
                                    const float min_coord,
                                    const float range,
                                    const int n_brt_nodes) {
  // do the initial setup on 1 thread
  if (threadIdx.x == 0) {
    const auto root_level = rt_prefix_n[0] / 3;
    const auto root_prefix = codes[0] >> (morton_bits - (3 * root_level));

    // compute root's corner
    shared::morton32_to_xyz(&u_corner[0],
                            root_prefix << (morton_bits - (3 * root_level)),
                            min_coord,
                            range);
    u_cell_size[0] = range;
  }

  __syncthreads();

  const auto n = static_cast<unsigned>(n_brt_nodes);
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  // all threads participate in the main work
  // i > 0 && i < N
  for (auto i = idx; i < n; i += stride) {
    if (i == 0) {
      continue;
    }
    shared::v2::ProcessOctNode(static_cast<int>(i),
                               u_children,
                               u_corner,
                               u_cell_size,
                               u_child_node_mask,
                               node_offsets,
                               node_counts,
                               codes,
                               rt_prefix_n,
                               rt_parents,
                               min_coord,
                               range);
  }
}

__global__ void gpu::k_LinkLeafNodes(int (*u_children)[8],
                                     int* u_child_leaf_mask,
                                     const int* node_offsets,
                                     const int* node_counts,
                                     const unsigned int* codes,
                                     const bool* rt_has_leaf_left,
                                     const bool* rt_has_leaf_right,
                                     const uint8_t* rt_prefix_n,
                                     const int* rt_parents,
                                     const int* rt_left_child,
                                     const int n_brt_nodes) {
  const auto n = static_cast<unsigned int>(n_brt_nodes);
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  // all threads participate in the main work
  // i < N
  for (auto i = idx; i < n; i += stride) {
    shared::v2::ProcessLinkLeaf(static_cast<int>(i),
                                u_children,
                                u_child_leaf_mask,
                                node_offsets,
                                node_counts,
                                codes,
                                rt_has_leaf_left,
                                rt_has_leaf_right,
                                rt_prefix_n,
                                rt_parents,
                                rt_left_child);
  }
}

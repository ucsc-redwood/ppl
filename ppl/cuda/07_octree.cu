#include <device_launch_parameters.h>

#include "shared/oct_v2.h"

namespace gpu {

__global__ void k_MakeOctNodes(int (*u_children)[8],
                               glm::vec4* u_corner,
                               float* u_cell_size,
                               int* u_child_node_mask,
                               const int* node_offsets,  // prefix sum
                               const int* node_counts,   // edge count
                               const unsigned int* codes,
                               const uint8_t* rt_prefixN,
                               const int* rt_parents,
                               const float min_coord,
                               const float range,
                               const int n_brt_nodes) {
  // do the initial setup on 1 thread
  if (threadIdx.x == 0) {
    const auto root_level = rt_prefixN[0] / 3;
    const auto root_prefix = codes[0] >> (morton_bits - (3 * root_level));

    // compute root's corner
    shared::morton32_to_xyz(&u_corner[0],
                            root_prefix << (morton_bits - (3 * root_level)),
                            min_coord,
                            range);
    u_cell_size[0] = range;
  }

  __syncthreads();

  const auto N = n_brt_nodes;
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  // all threads participate in the main work
  // i > 0 && i < N
  for (auto i = idx; i < N; i += stride) {
    if (i == 0) {
      continue;
    }
    shared::v2::ProcessOctNode(i,
                               u_children,
                               u_corner,
                               u_cell_size,
                               u_child_node_mask,
                               node_offsets,
                               node_counts,
                               codes,
                               rt_prefixN,
                               rt_parents,
                               min_coord,
                               range);
  }
}

__global__ void k_LinkLeafNodes(
    // --- new parameters
    int (*u_children)[8],
    int* u_child_leaf_mask,
    // --- end new parameters
    const int* node_offsets,
    const int* node_counts,
    const unsigned int* codes,
    const bool* rt_hasLeafLeft,
    const bool* rt_hasLeafRight,
    const uint8_t* rt_prefixN,
    const int* rt_parents,
    const int* rt_leftChild,
    const int n_brt_nodes) {
  const auto N = n_brt_nodes;
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  // all threads participate in the main work
  // i < N
  for (auto i = idx; i < N; i += stride) {
    shared::v2::ProcessLinkLeaf(i,
                                u_children,
                                u_child_leaf_mask,
                                node_offsets,
                                node_counts,
                                codes,
                                rt_hasLeafLeft,
                                rt_hasLeafRight,
                                rt_prefixN,
                                rt_parents,
                                rt_leftChild);
  }
}

// -----------------------------------------------------------------------------
// Debug Kernels
// -----------------------------------------------------------------------------

namespace debug {

// Use 1 thread to compute the cpu's code
__global__ void k_MakeOctNodes(int (*u_children)[8],
                               glm::vec4* u_corner,
                               float* u_cell_size,
                               int* u_child_node_mask,
                               const int* node_offsets,  // prefix sum
                               const int* node_counts,   // edge count
                               const unsigned int* codes,
                               const uint8_t* rt_prefixN,
                               const int* rt_parents,
                               const float min_coord,
                               const float range,
                               const int n_brt_nodes) {
  // do the initial setup on 1 thread
  if (threadIdx.x == 0) {
    // Computing the first node
    const auto root_level = rt_prefixN[0] / 3;
    const auto root_prefix = codes[0] >> (morton_bits - (3 * root_level));

    glm::vec4 ret;
    shared::morton32_to_xyz(&ret,
                            root_prefix << (morton_bits - (3 * root_level)),
                            min_coord,
                            range);
    u_corner[0] = ret;
    u_cell_size[0] = range;

    // Skipping the first node
    for (auto i = 1; i < n_brt_nodes; i++) {
      if (node_counts[i] > 0) {
        shared::v2::ProcessOctNode(i,
                                   u_children,
                                   u_corner,
                                   u_cell_size,
                                   u_child_node_mask,
                                   node_offsets,
                                   node_counts,
                                   codes,
                                   rt_prefixN,
                                   rt_parents,
                                   min_coord,
                                   range);
      }
    }
  }
}

}  // namespace debug

}  // namespace gpu
#include "openmp/kernels/07_octree.hpp"

#include "shared/oct_v2.h"

namespace cpu {

void k_MakeOctNodes(int (*u_children)[8],
                    glm::vec4* u_corner,
                    float* u_cell_size,
                    int* u_child_node_mask,
                    const int* node_offsets,
                    const int* node_counts,
                    const unsigned int* codes,
                    const uint8_t* rt_prefixN,
                    const int* rt_parents,
                    const float min_coord,
                    const float range,
                    const int n_brt_nodes) {
  // Computing the first node
  const auto root_level = rt_prefixN[0] / 3;
  const auto root_prefix = codes[0] >> (morton_bits - (3 * root_level));

  glm::vec4 ret;
  shared::morton32_to_xyz(
      &ret, root_prefix << (morton_bits - (3 * root_level)), min_coord, range);
  u_corner[0] = ret;
  u_cell_size[0] = range;

  // Skipping the first node
#pragma omp parallel for
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

void k_LinkLeafNodes(int (*u_children)[8],
                     int* u_child_leaf_mask,
                     const int* node_offsets,
                     const int* node_counts,
                     const unsigned int* codes,
                     const bool* rt_hasLeafLeft,
                     const bool* rt_hasLeafRight,
                     const uint8_t* rt_prefixN,
                     const int* rt_parents,
                     const int* rt_leftChild,
                     const int n_brt_nodes) {
#pragma omp parallel for
  for (auto i = 0; i < n_brt_nodes; i++) {
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
}  // namespace cpu
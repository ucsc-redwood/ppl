#pragma once

#include <glm/glm.hpp>

namespace gpu {

__global__ void k_MakeOctNodes(
    // --- octree parameters
    int (*u_children)[8],
    glm::vec4* u_corner,
    float* u_cell_size,
    int* u_child_node_mask,
    // --- end octree parameters
    const int* node_offsets,  // prefix sum
    const int* node_counts,   // edge count
    const unsigned int* codes,
    const uint8_t* rt_prefix_n,
    const int* rt_parents,
    float min_coord,
    float range,
    int n_brt_nodes);

__global__ void k_LinkLeafNodes(int (*u_children)[8],
                                int* u_child_leaf_mask,
                                const int* node_offsets,
                                const int* node_counts,
                                const unsigned int* codes,
                                const bool* rt_has_leaf_left,
                                const bool* rt_has_leaf_right,
                                const uint8_t* rt_prefix_n,
                                const int* rt_parents,
                                const int* rt_left_child,
                                int n_brt_nodes);

}  // namespace gpu
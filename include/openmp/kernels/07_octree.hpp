#pragma once

#include <cuda_runtime_api.h>

namespace cpu {

__global__ void k_MakeOctNodes(
    // --- new parameters
    int (*u_children)[8],
    glm::vec4* u_corner,
    float* u_cell_size,
    int* u_child_node_mask,
    // --- end new parameters
    const int* node_offsets,    // prefix sum
    const int* rt_node_counts,  // edge count
    const unsigned int* codes,
    const uint8_t* rt_prefixN,
    const int* rt_parents,
    const float min_coord,
    const float range,
    const int n_brt_nodes);

__global__ void k_LinkLeafNodes(
    // --- new parameters
    int (*u_children)[8],
    int* u_child_leaf_mask,
    // --- end new parameters
    const int* node_offsets,
    const int* rt_node_counts,
    const unsigned int* codes,
    const bool* rt_hasLeafLeft,
    const bool* rt_hasLeafRight,
    const uint8_t* rt_prefixN,
    const int* rt_parents,
    const int* rt_leftChild,
    const int n_brt_nodes);

}  // namespace cpu
#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace cpu {

void k_MakeOctNodes(int n_threads,
                    // --- new parameters
                    int (*u_children)[8],
                    glm::vec4* u_corner,
                    float* u_cell_size,
                    int* u_child_node_mask,
                    // --- end new parameters
                    const int* node_offsets,  // prefix sum
                    const int* node_counts,   // edge count
                    const unsigned int* codes,
                    const uint8_t* rt_prefixN,
                    const int* rt_parents,
                    float min_coord,
                    float range,
                    int n_brt_nodes);

void k_LinkLeafNodes(int n_threads,
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
                     int n_brt_nodes);

}  // namespace cpu

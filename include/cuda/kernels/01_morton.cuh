#pragma once

#include <glm/glm.hpp>

namespace gpu {

// ============================================================================
// Kernel entry points
// ============================================================================

/**
 * @brief Compute the 3D morton code for each point in the input data.
 *
 * @param data The input data.
 * @param morton_keys The output morton codes.
 * @param n The number of points in the input data.
 * @param min_coord The minimum coordinate value in the input data.
 * @param range The range of the input data.
 */
__global__ void k_ComputeMortonCode(const glm::vec4* data,
                                    unsigned int* morton_keys,
                                    size_t n,
                                    float min_coord,
                                    float range);

}  // namespace gpu

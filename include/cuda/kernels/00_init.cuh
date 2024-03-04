#pragma once

#include <glm/glm.hpp>

namespace gpu {

// ============================================================================
// Kernel entry points
// ============================================================================

__global__ void k_InitRandomVec4(
    glm::vec4 *u_data, int n, float min, float range, int seed);

}  // namespace gpu
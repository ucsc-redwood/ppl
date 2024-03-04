#pragma once

#include <glm/glm.hpp>

namespace cpu {

void k_InitRandomVec4(
    glm::vec4 *u_data, int n, float min, float range, int seed);

}  // namespace cpu
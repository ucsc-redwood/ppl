#pragma once

#include <glm/glm.hpp>

namespace cpu {

void k_ComputeMortonCode(const glm::vec4* data,
                         unsigned int* morton_keys,
                         int n,
                         float min_coord,
                         float range);

}  // namespace cpu

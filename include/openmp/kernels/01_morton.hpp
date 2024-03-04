#pragma once

#include <glm/glm.hpp>

namespace cpu {

void k_ComputeMortonCode(const glm::vec4* data,
                         unsigned int* morton_keys,
                         size_t n,
                         float min_coord,
                         float range);

}  // namespace cpu

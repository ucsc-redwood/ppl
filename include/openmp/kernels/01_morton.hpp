#pragma once

#include <glm/glm.hpp>

namespace cpu {

void k_ComputeMortonCode(const int thread_start,
                         const int thread_end,
                         const glm::vec4* data,
                         unsigned int* morton_keys,
                         const int n,
                         const float min_coord,
                         const float range);

}  // namespace cpu


#include <omp.h>

#include <glm/glm.hpp>

#include "shared/morton.h"

namespace cpu {

void k_ComputeMortonCode(const glm::vec4* data,
                         unsigned int* morton_keys,
                         size_t n,
                         float min_coord,
                         float range) {
#pragma omp parallel for
  for (auto i = 0; i < n; i++) {
    morton_keys[i] = shared::xyz_to_morton32(data[i], min_coord, range);
  }
}

}  // namespace cpu
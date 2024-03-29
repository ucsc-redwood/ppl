#include <glm/glm.hpp>

#include "shared/morton.h"

namespace cpu {

void k_ComputeMortonCode(const int n_threads,
                         const glm::vec4* data,
                         unsigned int* morton_keys,
                         const int n,
                         const float min_coord,
                         const float range) {
#pragma omp parallel for num_threads(n_threads)
  for (auto i = 0; i < n; i++) {
    morton_keys[i] = shared::xyz_to_morton32(data[i], min_coord, range);
  }
}

}  // namespace cpu

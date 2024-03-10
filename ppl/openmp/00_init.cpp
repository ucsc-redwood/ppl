#include <algorithm>
#include <glm/glm.hpp>
#include <random>

namespace cpu {

void k_InitRandomVec4(glm::vec4 *u_data,
                      const int n,
                      const float min,
                      const float range,
                      const int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution dis(min, min + range);
  std::generate_n(
      u_data, n, [&] { return glm::vec4(dis(gen), dis(gen), dis(gen), 0.0f); });
}

}  // namespace cpu

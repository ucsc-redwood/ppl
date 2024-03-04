#include <algorithm>
#include <glm/glm.hpp>
#include <random>

namespace cpu {

void k_InitRandomVec4(
    glm::vec4 *u_data, int n, float min, float range, int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dis(min, min + range);
  std::generate(u_data, u_data + n, [&] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 0.0f);
  });
}

}  // namespace cpu

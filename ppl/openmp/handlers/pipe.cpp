#include "handlers/pipe.h"

#include <spdlog/spdlog.h>

#include <glm/glm.hpp>

Pipe::Pipe(const size_t n,
           const float min_coord,
           const float range,
           const int seed)
    : n(n),
      n_unique_keys(),
      n_brt_nodes(),
      n_oct_nodes(),
      min_coord(min_coord),
      range(range),
      seed(seed),
      sort(n),
      unique(n),
      brt(n),
      oct(static_cast<size_t>(static_cast<double>(n) * EDUCATED_GUESS)) {
  u_points = new glm::vec4[n];
  u_edge_count = new int[n];
  u_edge_offset = new int[n];

  spdlog::trace("On constructor: Pipe, n: {}", n);
}

Pipe::~Pipe() {
  delete[] u_points;
  delete[] u_edge_count;
  delete[] u_edge_offset;

  spdlog::trace("On destructor: Pipe");
}

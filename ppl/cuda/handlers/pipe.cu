#include <spdlog/spdlog.h>

#include <glm/glm.hpp>

#include "cuda/helper.cuh"
#include "handlers/pipe.h"

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
  MALLOC_MANAGED(&u_points, n);
  MALLOC_MANAGED(&u_edge_count, n);
  MALLOC_MANAGED(&u_edge_offset, n);

  SYNC_DEVICE();

  spdlog::trace("On constructor: Pipe, n: {}", n);
}

Pipe::~Pipe() {
  CUDA_FREE(u_points);
  CUDA_FREE(u_edge_count);
  CUDA_FREE(u_edge_offset);

  spdlog::trace("On destructor: Pipe");
}

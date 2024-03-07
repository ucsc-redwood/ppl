#include <cstdint>

#include "shared/edge_count.h"

namespace cpu {

void k_EdgeCount(const int n_threads,
                 const uint8_t* prefix_n,
                 const int* parents,
                 int* edge_count,
                 const int n_brt_nodes) {
  // skipping first
#pragma omp parallel for num_threads(n_threads)
  for (int i = 1; i < n_brt_nodes; i++) {
    shared::processEdgeCount(i, prefix_n, parents, edge_count);
  }

  edge_count[0] = 0;
}

}  // namespace cpu
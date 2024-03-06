#include <cstdint>

namespace cpu {

void k_EdgeCount(const uint8_t* prefix_n,
                 const int* parents,
                 int* edge_count,
                 const int n_brt_nodes) {}

void k_EdgeCount(const int n_threads,
                 const uint8_t* prefix_n,
                 const int* parents,
                 int* edge_count,
                 const int n_brt_nodes) {
  edge_count[0] = 0;
  // skipping first

#pragma omp parallel for num_threads(n_threads)
  for (int i = 1; i < n_brt_nodes; i++) {
    const auto my_depth = prefix_n[i] / 3;
    const auto parent_depth = prefix_n[parents[i]] / 3;
    edge_count[i] = my_depth - parent_depth;
  }
}

}  // namespace cpu
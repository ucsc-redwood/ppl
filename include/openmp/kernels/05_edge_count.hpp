#pragma once

#include <cstdint>

namespace cpu {

void k_EdgeCount(int n_threads,
                 const uint8_t* prefix_n,
                 const int* parents,
                 int* edge_count,
                 int n_brt_nodes);

}  // namespace cpu
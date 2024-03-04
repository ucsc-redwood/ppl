#pragma once

namespace cpu {

// ============================================================================
// Kernel entry points
// ============================================================================

void k_EdgeCount(const uint8_t* prefix_n,
                 const int* parents,
                 int* edge_count,
                 int n_brt_nodes);

}  // namespace cpu
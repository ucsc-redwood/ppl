#pragma once

#include <spdlog/spdlog.h>

#include <numeric>

#include "cuda/helper.cuh"
#include "cuda/kernels/04_radix_tree.cuh"

namespace gpu {

inline void dispatch_BuildRadixTree(const int grid_size,
                                    const cudaStream_t stream,
                                    const unsigned int* codes,
                                    uint8_t* prefix_n,
                                    bool* has_leaf_left,
                                    bool* has_leaf_right,
                                    int* left_child,
                                    int* parent,
                                    const int n_unique) {
  constexpr auto n_threads = 512;

  spdlog::debug("Dispatching k_BuildRadixTree with ({} blocks, {} threads)",
                grid_size,
                n_threads);

  k_BuildRadixTree<<<grid_size, n_threads, 0, stream>>>(n_unique,
                                                        codes,
                                                        prefix_n,
                                                        has_leaf_left,
                                                        has_leaf_right,
                                                        left_child,
                                                        parent);
}

}  // namespace gpu
#include <spdlog/spdlog.h>

#include "cuda/helper.cuh"
#include "handlers/radix_tree.h"

// Let's allocate 'n' instead of 'n_brt_nodes' for now
RadixTree::RadixTree(const size_t n) : n(n), n_brt_nodes() {
  MALLOC_MANAGED(&u_prefix_n, n);
  MALLOC_MANAGED(&u_has_leaf_left, n);
  MALLOC_MANAGED(&u_has_leaf_right, n);
  MALLOC_MANAGED(&u_left_child, n);
  MALLOC_MANAGED(&u_parent, n);

  SYNC_DEVICE();

  spdlog::trace("On constructor: RadixTree, n: {}", n);
}

RadixTree::~RadixTree() {
  CUDA_FREE(u_prefix_n);
  CUDA_FREE(u_has_leaf_left);
  CUDA_FREE(u_has_leaf_right);
  CUDA_FREE(u_left_child);
  CUDA_FREE(u_parent);

  spdlog::trace("On destructor: RadixTree");
}

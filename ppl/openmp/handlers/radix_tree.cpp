#include "handlers/radix_tree.h"

#include <spdlog/spdlog.h>

// Let's allocate 'n' instead of 'n_brt_nodes' for now
// Because usually n_brt_nodes is 99.x% of n

RadixTree::RadixTree(const size_t n) : n(n), n_brt_nodes() {
  u_prefix_n = new uint8_t[n];
  u_has_leaf_left = new bool[n];
  u_has_leaf_right = new bool[n];
  u_left_child = new int[n];
  u_parent = new int[n];

  spdlog::trace("On constructor: RadixTree, n: {}", n);
}

RadixTree::~RadixTree() {
  delete[] u_prefix_n;
  delete[] u_has_leaf_left;
  delete[] u_has_leaf_right;
  delete[] u_left_child;
  delete[] u_parent;

  spdlog::trace("On destructor: RadixTree");
}

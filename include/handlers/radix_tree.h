#pragma once

#include <cstddef>
#include <cstdint>

struct RadixTree {
  // ------------------------
  // Essential Data
  // ------------------------
  const size_t n;
  size_t n_brt_nodes;

  uint8_t* u_prefix_n;
  bool* u_has_leaf_left;
  bool* u_has_leaf_right;
  int* u_left_child;
  int* u_parent;

  // ------------------------

  RadixTree() = delete;

  // Let's allocate 'n' instead of 'n_brt_nodes' for now
  explicit RadixTree(const size_t n);

  RadixTree(const RadixTree&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree(RadixTree&&) = delete;
  RadixTree& operator=(RadixTree&&) = delete;

  ~RadixTree();

  [[nodiscard]] size_t size() const { return n; }
};

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

  // void setNumBrtNodes(const size_t n_brt_nodes) {
  //   this->n_brt_nodes = n_brt_nodes;
  // }

  // [[nodiscard]] size_t getNumBrtNodes() const { return n_brt_nodes; }

  // void attachStreamSingle(const cudaStream_t stream);
  // void attachStreamHost(const cudaStream_t stream);
};

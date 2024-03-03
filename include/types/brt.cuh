#pragma once

#include "cuda/unified_vector.cuh"

struct RadixTree {
  explicit RadixTree(const size_t n)
      : u_prefix_n(n), u_left_child(n), u_parent(n) {
    // Well, you know you can't really use vector of bool in C++, right?
    MallocManaged(&u_has_leaf_left, n);
    MallocManaged(&u_has_leaf_right, n);
  }

  ~RadixTree() {
    CUDA_FREE(u_has_leaf_left);
    CUDA_FREE(u_has_leaf_right);
  }

  RadixTree(const RadixTree&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree(RadixTree&&) = delete;
  RadixTree& operator=(RadixTree&&) = delete;

  void attachStream(const cudaStream_t stream) {
    ATTACH_STREAM_SINGLE(u_prefix_n.data());
    ATTACH_STREAM_SINGLE(u_has_leaf_left);
    ATTACH_STREAM_SINGLE(u_has_leaf_right);
    ATTACH_STREAM_SINGLE(u_left_child.data());
    ATTACH_STREAM_SINGLE(u_parent.data());
  }

  [[nodiscard]] size_t getMemorySize() const {
    size_t total = 0;
    total += calculateMemorySize(u_prefix_n);
    const auto n = u_prefix_n.size();
    total += sizeof(bool) * n;
    total += sizeof(bool) * n;
    total += calculateMemorySize(u_left_child);
    total += calculateMemorySize(u_parent);
    return total;
  }

  cu::unified_vector<int> u_prefix_n;
  bool* u_has_leaf_left;
  bool* u_has_leaf_right;
  cu::unified_vector<int> u_left_child;
  cu::unified_vector<int> u_parent;
};

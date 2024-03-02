#pragma once

#include "cuda/unified_vector.cuh"

struct RadixTree {
  explicit RadixTree(const size_t n)
      : u_prefixN(n), u_leftChild(n), u_parent(n) {
    // Well, you know you can't really use vector of bools in C++, right?
    MallocManaged(&u_hasLeafLeft, n);
    MallocManaged(&u_hasLeafRight, n);
  }

  ~RadixTree() {
    CUDA_FREE(u_hasLeafLeft);
    CUDA_FREE(u_hasLeafRight);
  }

  RadixTree(const RadixTree&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree(RadixTree&&) = delete;
  RadixTree& operator=(RadixTree&&) = delete;

  void attachStream(const cudaStream_t stream) {
    ATTACH_STREAM_SINGLE(u_prefixN.data());
    ATTACH_STREAM_SINGLE(u_hasLeafLeft);
    ATTACH_STREAM_SINGLE(u_hasLeafRight);
    ATTACH_STREAM_SINGLE(u_leftChild.data());
    ATTACH_STREAM_SINGLE(u_parent.data());
  }

  [[nodiscard]] size_t getMemorySize() const {
    size_t total = 0;
    total += calculateMemorySize(u_prefixN);
    const auto n = u_prefixN.size();
    total += sizeof(bool) * n;
    total += sizeof(bool) * n;
    total += calculateMemorySize(u_leftChild);
    total += calculateMemorySize(u_parent);
    return total;
  }

  cu::unified_vector<int> u_prefixN;
  bool* u_hasLeafLeft;
  bool* u_hasLeafRight;
  cu::unified_vector<int> u_leftChild;
  cu::unified_vector<int> u_parent;
};

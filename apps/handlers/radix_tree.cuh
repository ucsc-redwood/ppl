#pragma once

#include "cuda/helper.cuh"
#include "cuda/kernels/04_radix_tree.cuh"

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
  explicit RadixTree(const size_t n) : n(n) {
    MALLOC_MANAGED(&u_prefix_n, n);
    MALLOC_MANAGED(&u_has_leaf_left, n);
    MALLOC_MANAGED(&u_has_leaf_right, n);
    MALLOC_MANAGED(&u_left_child, n);
    MALLOC_MANAGED(&u_parent, n);

    SYNC_DEVICE();

    spdlog::trace("On constructor: RadixTree, n: {}", n);
  }

  RadixTree(const RadixTree&) = delete;
  RadixTree& operator=(const RadixTree&) = delete;
  RadixTree(RadixTree&&) = delete;
  RadixTree& operator=(RadixTree&&) = delete;

  ~RadixTree() {
    CUDA_FREE(u_prefix_n);
    CUDA_FREE(u_has_leaf_left);
    CUDA_FREE(u_has_leaf_right);
    CUDA_FREE(u_left_child);
    CUDA_FREE(u_parent);

    spdlog::trace("On destructor: RadixTree");
  }

  [[nodiscard]] size_t size() const { return n; }

  void setNumBrtNodes(const size_t n_brt_nodes) {
    this->n_brt_nodes = n_brt_nodes;
  }
  [[nodiscard]] size_t getNumBrtNodes() const { return n_brt_nodes; }

  void attachStreamSingle(const cudaStream_t stream) const {
    ATTACH_STREAM_SINGLE(u_prefix_n);
    ATTACH_STREAM_SINGLE(u_has_leaf_left);
    ATTACH_STREAM_SINGLE(u_has_leaf_right);
    ATTACH_STREAM_SINGLE(u_left_child);
    ATTACH_STREAM_SINGLE(u_parent);
  }

  void attachStreamHost(const cudaStream_t stream) const {
    ATTACH_STREAM_HOST(u_prefix_n);
    ATTACH_STREAM_HOST(u_has_leaf_left);
    ATTACH_STREAM_HOST(u_has_leaf_right);
    ATTACH_STREAM_HOST(u_left_child);
    ATTACH_STREAM_HOST(u_parent);
    SYNC_STREAM(stream);
  }
};

namespace gpu {
namespace v2 {

static void dispatch_BuildRadixTree(const int grid_size,
                                    const cudaStream_t stream,
                                    const unsigned int* u_unique_morton_keys,
                                    const size_t n_unique_keys,
                                    RadixTree& radix_tree) {
  constexpr auto n_threads = 512;

  spdlog::debug("Dispatching k_BuildRadixTree with ({} blocks, {} threads)",
                grid_size,
                n_threads);

  gpu::k_BuildRadixTree<<<grid_size, n_threads, 0, stream>>>(
      n_unique_keys,
      u_unique_morton_keys,
      radix_tree.u_prefix_n,
      radix_tree.u_has_leaf_left,
      radix_tree.u_has_leaf_right,
      radix_tree.u_left_child,
      radix_tree.u_parent);
}
}  // namespace v2
}  // namespace gpu
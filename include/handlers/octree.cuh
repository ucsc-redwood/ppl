#pragma once

#include <spdlog/spdlog.h>

#include "radix_tree.cuh"

struct OctreeHandler {
  // ------------------------
  // Essential Data
  // ------------------------

  const size_t n_octr_nodes_to_allocate;
  size_t n_oct_nodes;

  // Output:

  int (*u_children)[8];
  glm::vec4* u_corner;
  float* u_cell_size;
  int* u_child_node_mask;
  int* u_child_leaf_mask;

  // ------------------------

  OctreeHandler() = delete;

  explicit OctreeHandler(const size_t n_octr_nodes_to_allocate)
      : n_octr_nodes_to_allocate(n_octr_nodes_to_allocate) {
    // MALLOC_MANAGED(&u_children, n_octr_nodes_to_allocate * 8);

    CHECK_CUDA_CALL(cudaMallocManaged(
        &u_children, n_octr_nodes_to_allocate * 8 * sizeof(int)));

    MALLOC_MANAGED(&u_corner, n_octr_nodes_to_allocate);
    MALLOC_MANAGED(&u_cell_size, n_octr_nodes_to_allocate);
    MALLOC_MANAGED(&u_child_node_mask, n_octr_nodes_to_allocate);
    MALLOC_MANAGED(&u_child_leaf_mask, n_octr_nodes_to_allocate);
    SYNC_DEVICE();

    spdlog::trace("On constructor: OctreeHandler, n: {}",
                  n_octr_nodes_to_allocate);
  }

  OctreeHandler(const OctreeHandler&) = delete;
  OctreeHandler& operator=(const OctreeHandler&) = delete;
  OctreeHandler(OctreeHandler&&) = delete;
  OctreeHandler& operator=(OctreeHandler&&) = delete;

  ~OctreeHandler() {
    CUDA_FREE(u_children);
    CUDA_FREE(u_corner);
    CUDA_FREE(u_cell_size);
    CUDA_FREE(u_child_node_mask);
    CUDA_FREE(u_child_leaf_mask);

    spdlog::trace("On destructor: OctreeHandler");
  }

  void attachStreamSingle(const cudaStream_t stream) const {
    ATTACH_STREAM_SINGLE(u_children);
    ATTACH_STREAM_SINGLE(u_corner);
    ATTACH_STREAM_SINGLE(u_cell_size);
    ATTACH_STREAM_SINGLE(u_child_node_mask);
    ATTACH_STREAM_SINGLE(u_child_leaf_mask);
  }

  void attachStreamGlobal(const cudaStream_t stream) const {
    ATTACH_STREAM_GLOBAL(u_children);
    ATTACH_STREAM_GLOBAL(u_corner);
    ATTACH_STREAM_GLOBAL(u_cell_size);
    ATTACH_STREAM_GLOBAL(u_child_node_mask);
    ATTACH_STREAM_GLOBAL(u_child_leaf_mask);
  }

  void attachStreamHost(const cudaStream_t stream) const {
    ATTACH_STREAM_HOST(u_children);
    ATTACH_STREAM_HOST(u_corner);
    ATTACH_STREAM_HOST(u_cell_size);
    ATTACH_STREAM_HOST(u_child_node_mask);
    ATTACH_STREAM_HOST(u_child_leaf_mask);
    SYNC_STREAM(stream);
  }
};

namespace gpu {
namespace v2 {

static void dispatch_BuildOctree(const int grid_size,
                                 const cudaStream_t stream,
                                 const RadixTree& brt,
                                 const unsigned int* u_sorted_keys,
                                 const int* u_edge_offset,
                                 const int* u_edge_count,
                                 OctreeHandler& oct,
                                 const float min_coord,
                                 const float range) {
  constexpr auto block_size = 512;

  spdlog::debug(
      "Dispatching k_MakeOctNodes with ({} blocks, {} threads) "
      "on {} items",
      grid_size,
      block_size,
      brt.getNumBrtNodes());

  k_MakeOctNodes<<<grid_size, block_size, 0, stream>>>(oct.u_children,
                                                       oct.u_corner,
                                                       oct.u_cell_size,
                                                       oct.u_child_node_mask,
                                                       u_edge_offset,
                                                       u_edge_count,
                                                       u_sorted_keys,
                                                       brt.u_prefix_n,
                                                       brt.u_parent,
                                                       min_coord,
                                                       range,
                                                       brt.getNumBrtNodes());
}

static void dispatch_LinkOctreeNodes(const int grid_size,
                                     const cudaStream_t stream,
                                     const int* node_offsets,
                                     const int* node_counts,
                                     const unsigned int* u_sorted_keys,
                                     const RadixTree& brt,
                                     OctreeHandler& oct) {
  constexpr auto block_size = 512;

  spdlog::debug(
      "Dispatching k_LinkLeafNodes with ({} blocks, {} threads) "
      "on {} items",
      grid_size,
      block_size,
      brt.getNumBrtNodes());

  k_LinkLeafNodes<<<grid_size, block_size, 0, stream>>>(oct.u_children,
                                                        oct.u_child_leaf_mask,
                                                        node_offsets,
                                                        node_counts,
                                                        u_sorted_keys,
                                                        brt.u_has_leaf_left,
                                                        brt.u_has_leaf_right,
                                                        brt.u_prefix_n,
                                                        brt.u_parent,
                                                        brt.u_left_child,
                                                        brt.getNumBrtNodes());
}

}  // namespace v2
}  // namespace gpu
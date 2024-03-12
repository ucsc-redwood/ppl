#pragma once

#include <glm/glm.hpp>

#include "radix_tree.h"

struct OctreeHandler {
  // ------------------------
  // Essential Data
  // ------------------------

  const size_t n_octr_nodes_to_allocate;
  size_t n_oct_nodes;

  // [Outputs]
  int (*u_children)[8];
  glm::vec4* u_corner;
  float* u_cell_size;
  int* u_child_node_mask;
  int* u_child_leaf_mask;

  // ------------------------

  OctreeHandler() = delete;

  explicit OctreeHandler(const size_t n_octr_nodes_to_allocate);

  OctreeHandler(const OctreeHandler&) = delete;
  OctreeHandler& operator=(const OctreeHandler&) = delete;
  OctreeHandler(OctreeHandler&&) = delete;
  OctreeHandler& operator=(OctreeHandler&&) = delete;

  ~OctreeHandler();
};

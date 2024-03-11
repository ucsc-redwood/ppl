#include "handlers/octree.h"

#include <spdlog/spdlog.h>

OctreeHandler::OctreeHandler(const size_t n_octr_nodes_to_allocate)
    : n_octr_nodes_to_allocate(n_octr_nodes_to_allocate), n_oct_nodes() {
  u_children = new int[n_octr_nodes_to_allocate][8];
  u_corner = new glm::vec4[n_octr_nodes_to_allocate];
  u_cell_size = new float[n_octr_nodes_to_allocate];
  u_child_node_mask = new int[n_octr_nodes_to_allocate];
  u_child_leaf_mask = new int[n_octr_nodes_to_allocate];

  spdlog::trace("On constructor: OctreeHandler, n: {}",
                n_octr_nodes_to_allocate);
}

OctreeHandler::~OctreeHandler() {
  delete[] u_children;
  delete[] u_corner;
  delete[] u_cell_size;
  delete[] u_child_node_mask;
  delete[] u_child_leaf_mask;

  spdlog::trace("On destructor: OctreeHandler");
}

#include "host_dispatcher.h"
#include "openmp/kernels/01_morton.hpp"
#include "openmp/kernels/02_sort.hpp"
#include "openmp/kernels/03_unique.hpp"
#include "openmp/kernels/04_radix_tree.hpp"
#include "openmp/kernels/05_edge_count.hpp"
#include "openmp/kernels/06_prefix_sum.hpp"
#include "openmp/kernels/07_octree.hpp"

void cpu::v2::dispatch_ComputeMorton(const int n_threads, const Pipe& pipe) {
  k_ComputeMortonCode(0,
                      n_threads - 1,
                      pipe.u_points,
                      pipe.sort.u_sort,
                      static_cast<int>(pipe.getInputSize()),
                      pipe.min_coord,
                      pipe.range);
}

void cpu::v2::dispatch_RadixSort(const int n_threads, const Pipe& pipe) {
  k_Sort(
      n_threads, pipe.sort.u_sort, pipe.sort.u_sort_alt, pipe.getInputSize());
}

void cpu::v2::dispatch_RemoveDuplicates([[maybe_unused]] int n_threads,
                                        Pipe& pipe) {
  std::copy_n(pipe.sort.u_sort, pipe.getInputSize(), pipe.unique.u_keys_out);

  const auto it = std::unique(pipe.unique.u_keys_out,
                              pipe.unique.u_keys_out + pipe.getInputSize());
  const auto n_unique = std::distance(pipe.unique.u_keys_out, it);
  pipe.n_unique_keys = n_unique;
  pipe.n_brt_nodes = n_unique - 1;
}

void cpu::v2::dispatch_BuildRadixTree(const int n_threads, const Pipe& pipe) {
  k_BuildRadixTree(n_threads,
                   static_cast<int>(pipe.getUniqueSize()),
                   pipe.getUniqueKeys(),
                   pipe.brt.u_prefix_n,
                   pipe.brt.u_has_leaf_left,
                   pipe.brt.u_has_leaf_right,
                   pipe.brt.u_left_child,
                   pipe.brt.u_parent);
}

void cpu::v2::dispatch_EdgeCount(const int n_threads, const Pipe& pipe) {
  k_EdgeCount(n_threads,
              pipe.brt.u_prefix_n,
              pipe.brt.u_parent,
              pipe.u_edge_count,
              static_cast<int>(pipe.getBrtSize()));
}

void cpu::v2::dispatch_EdgeOffset([[maybe_unused]] int n_threads, Pipe& pipe) {
  std::exclusive_scan(pipe.u_edge_count,
                      pipe.u_edge_count + pipe.getBrtSize(),
                      pipe.u_edge_offset,
                      0);
  pipe.n_oct_nodes = pipe.u_edge_offset[pipe.getBrtSize() - 1];
}

void cpu::v2::dispatch_BuildOctree(const int n_threads, const Pipe& pipe) {
  k_MakeOctNodes(n_threads,
                 pipe.oct.u_children,
                 pipe.oct.u_corner,
                 pipe.oct.u_cell_size,
                 pipe.oct.u_child_node_mask,
                 pipe.getEdgeOffset(),
                 pipe.getEdgeCount(),
                 pipe.getUniqueKeys(),
                 pipe.brt.u_prefix_n,
                 pipe.brt.u_parent,
                 pipe.min_coord,
                 pipe.range,
                 static_cast<int>(pipe.getBrtSize()));

  k_LinkLeafNodes(n_threads,
                  pipe.oct.u_children,
                  pipe.oct.u_child_leaf_mask,
                  pipe.getEdgeOffset(),
                  pipe.getEdgeCount(),
                  pipe.getUniqueKeys(),
                  pipe.brt.u_has_leaf_left,
                  pipe.brt.u_has_leaf_right,
                  pipe.brt.u_prefix_n,
                  pipe.brt.u_parent,
                  pipe.brt.u_left_child,
                  static_cast<int>(pipe.getBrtSize()));
}
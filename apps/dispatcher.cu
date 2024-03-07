#include "dispatcher.cuh"

void gpu::v2::dispatch_BuildOctree(const int grid_size,
                                   const cudaStream_t stream,
                                   const Pipe& pipe) {
  constexpr auto block_size = 512;

  k_MakeOctNodes<<<grid_size, block_size, 0, stream>>>(
      pipe.oct.u_children,
      pipe.oct.u_corner,
      pipe.oct.u_cell_size,
      pipe.oct.u_child_node_mask,
      pipe.u_edge_offset,
      pipe.u_edge_count,
      pipe.unique.u_keys_out,
      pipe.brt.u_prefix_n,
      pipe.brt.u_parent,
      pipe.min_coord,
      pipe.range,
      pipe.brt.getNumBrtNodes());

  k_LinkLeafNodes(pipe.oct.u_children,
                  pipe.oct.u_child_node_mask,
                  pipe.u_edge_offset,
                  pipe.u_edge_count,
                  pipe.unique.u_keys_out,
                  pipe.brt.u_has_leaf_left,
                  pipe.brt.u_has_leaf_right,
                  pipe.brt.u_prefix_n,
                  pipe.brt.u_parent,
                  pipe.brt.u_left_child,
                  static_cast<int>(pipe.brt.getNumBrtNodes()));
}
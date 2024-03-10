#include "dispatcher.h"

void gpu::v2::dispatch_ComputeMorton(const int grid_size,
                                     const cudaStream_t stream,
                                     Pipe& pipe) {
  constexpr auto block_size = 768;

  spdlog::debug(
      "Dispatching k_ComputeMortonCode with ({} blocks, {} "
      "threads) "
      "on {} items",
      grid_size,
      block_size,
      pipe.n);

  k_ComputeMortonCode<<<grid_size, block_size, 0, stream>>>(
      pipe.u_points, pipe.sort.data(), pipe.n, pipe.min_coord, pipe.range);
}

void gpu::v2::dispatch_RadixSort(const int grid_size,
                                 const cudaStream_t stream,
                                 const Pipe& pipe) {
  const auto n = pipe.n;

  pipe.sort.clearMem();
  pipe.sort.attachStreamSingle(stream);

  spdlog::debug("Dispatching k_GlobalHistogram, grid_size: {}, block_size: {}",
                grid_size,
                OneSweepHandler::GLOBAL_HIST_THREADS);

  k_GlobalHistogram<<<grid_size,
                      OneSweepHandler::GLOBAL_HIST_THREADS,
                      0,
                      stream>>>(
      pipe.sort.u_sort, pipe.sort.im_storage.d_global_histogram, n);

  spdlog::debug("Dispatching k_Scan, grid_size: {}, block_size: {}",
                OneSweepHandler::RADIX_PASSES,
                OneSweepHandler::RADIX);

  k_Scan<<<OneSweepHandler::RADIX_PASSES, OneSweepHandler::RADIX, 0, stream>>>(
      pipe.sort.im_storage.d_global_histogram,
      pipe.sort.im_storage.d_first_pass_histogram,
      pipe.sort.im_storage.d_second_pass_histogram,
      pipe.sort.im_storage.d_third_pass_histogram,
      pipe.sort.im_storage.d_fourth_pass_histogram);

  spdlog::debug(
      "Dispatching k_DigitBinningPass, grid_size: {}, block_size: {} ... x4",
      grid_size,
      OneSweepHandler::BINNING_THREADS);

  k_DigitBinningPass<<<grid_size,
                       OneSweepHandler::BINNING_THREADS,
                       0,
                       stream>>>(pipe.sort.u_sort,  // <---
                                 pipe.sort.u_sort_alt,
                                 pipe.sort.im_storage.d_first_pass_histogram,
                                 pipe.sort.im_storage.d_index,
                                 n,
                                 0);

  k_DigitBinningPass<<<grid_size,
                       OneSweepHandler::BINNING_THREADS,
                       0,
                       stream>>>(pipe.sort.u_sort_alt,
                                 pipe.sort.u_sort,  // <---
                                 pipe.sort.im_storage.d_second_pass_histogram,
                                 pipe.sort.im_storage.d_index,
                                 n,
                                 8);

  k_DigitBinningPass<<<grid_size,
                       OneSweepHandler::BINNING_THREADS,
                       0,
                       stream>>>(pipe.sort.u_sort,  // <---
                                 pipe.sort.u_sort_alt,
                                 pipe.sort.im_storage.d_third_pass_histogram,
                                 pipe.sort.im_storage.d_index,
                                 n,
                                 16);

  k_DigitBinningPass<<<grid_size,
                       OneSweepHandler::BINNING_THREADS,
                       0,
                       stream>>>(pipe.sort.u_sort_alt,
                                 pipe.sort.u_sort,  // <---
                                 pipe.sort.im_storage.d_fourth_pass_histogram,
                                 pipe.sort.im_storage.d_index,
                                 n,
                                 24);
}

void gpu::v2::dispatch_RemoveDuplicates(int grid_size,
                                        cudaStream_t stream,
                                        Pipe& pipe) {
  constexpr auto n_threads = UniqueAgent::n_threads;  // 256

  spdlog::debug("Dispatching k_FindDups with ({} blocks, {} threads)",
                grid_size,
                n_threads);

  k_FindDups<<<grid_size, n_threads, 0, stream>>>(
      pipe.getSortedKeys(),
      pipe.unique.im_storage.u_flag_heads,
      pipe.getInputSize());

  SYNC_STREAM(stream);
  std::partial_sum(pipe.unique.im_storage.u_flag_heads,
                   pipe.unique.im_storage.u_flag_heads + pipe.getInputSize(),
                   pipe.unique.im_storage.u_flag_heads);

  spdlog::debug("Dispatching k_MoveDups with ({} blocks, {} threads)",
                grid_size,
                n_threads);

  k_MoveDups<<<grid_size, n_threads, 0, stream>>>(
      pipe.getSortedKeys(),
      pipe.unique.im_storage.u_flag_heads,
      pipe.getInputSize(),
      pipe.unique.u_keys_out,
      nullptr);

  SYNC_STREAM(stream);
  pipe.n_unique_keys =
      pipe.unique.im_storage.u_flag_heads[pipe.getInputSize() - 1];
  pipe.n_brt_nodes = pipe.n_unique_keys - 1;
}

void gpu::v2::dispatch_BuildRadixTree(const int grid_size,
                                      const cudaStream_t stream,
                                      const Pipe& pipe) {
  constexpr auto n_threads = 512;

  spdlog::debug("Dispatching k_BuildRadixTree with ({} blocks, {} threads)",
                grid_size,
                n_threads);

  k_BuildRadixTree<<<grid_size, n_threads, 0, stream>>>(
      pipe.getUniqueSize(),
      pipe.getUniqueKeys(),
      pipe.brt.u_prefix_n,
      pipe.brt.u_has_leaf_left,
      pipe.brt.u_has_leaf_right,
      pipe.brt.u_left_child,
      pipe.brt.u_parent);
}

void gpu::v2::dispatch_EdgeCount(const int grid_size,
                                 const cudaStream_t stream,
                                 const Pipe& pipe) {
  constexpr auto block_size = 512;

  spdlog::debug(
      "Dispatching k_EdgeCount with ({} blocks, {} threads) "
      "on {} items",
      grid_size,
      block_size,
      pipe.getBrtSize());

  k_EdgeCount<<<grid_size, block_size, 0, stream>>>(pipe.brt.u_prefix_n,
                                                    pipe.brt.u_parent,
                                                    pipe.u_edge_count,
                                                    pipe.getBrtSize());
}

void gpu::v2::dispatch_EdgeOffset([[maybe_unused]] const int grid_size,
                                  const cudaStream_t stream,
                                  const Pipe& pipe) {
  constexpr auto n_threads = PrefixSumAgent<int>::n_threads;

  spdlog::debug(
      "Dispatching k_SingleBlockExclusiveScan with (1 blocks, {} "
      "threads)",
      n_threads);

  k_SingleBlockExclusiveScan<<<1, n_threads, 0, stream>>>(
      pipe.getEdgeCount(), pipe.u_edge_offset, pipe.getBrtSize());
}

void gpu::v2::dispatch_BuildOctree(const int grid_size,
                                   const cudaStream_t stream,
                                   const Pipe& pipe) {
  constexpr auto block_size = 512;

  spdlog::debug(
      "dispatching k_MakeOctNodes: grid_size: {}, block_size: {}, on {} num of "
      "data",
      grid_size,
      block_size,
      pipe.getBrtSize());

  k_MakeOctNodes<<<grid_size, block_size, 0, stream>>>(
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
      pipe.getBrtSize());

  spdlog::debug(
      "dispatching k_LinkLeafNodes: grid_size: {}, block_size: {}, on {} num "
      "of data",
      grid_size,
      block_size,
      pipe.getBrtSize());

  k_LinkLeafNodes<<<grid_size, block_size, 0, stream>>>(
      pipe.oct.u_children,
      pipe.oct.u_child_node_mask,
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

#include <omp.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cub/cub.cuh>
#include <memory>

#include "app_params.hpp"
#include "kernels_fwd.h"
#include "naive_pipe.cuh"

void run_all_in_gpu(NaivePipe* pipe,
                    const AppParams& params,
                    const cudaStream_t stream) {
  gpu::dispatch_InitRandomVec4(params.n_blocks,
                               stream,
                               pipe->u_points.data(),
                               params.n,
                               params.min_coord,
                               params.getRange(),
                               params.seed);

  gpu::dispatch_ComputeMorton(params.n_blocks,
                              stream,
                              pipe->u_points.data(),
                              pipe->u_morton_keys.data(),
                              params.n,
                              params.min_coord,
                              params.getRange());

  gpu::dispatch_RadixSort(params.n_blocks,
                          stream,
                          pipe->u_morton_keys.data(),
                          pipe->sort_tmp.u_sort_alt.data(),
                          pipe->sort_tmp.u_global_histogram.data(),
                          pipe->sort_tmp.u_index.data(),
                          pipe->sort_tmp.u_first_pass_histogram.data(),
                          pipe->sort_tmp.u_second_pass_histogram.data(),
                          pipe->sort_tmp.u_third_pass_histogram.data(),
                          pipe->sort_tmp.u_fourth_pass_histogram.data(),
                          params.n);
  int num_unique;
  gpu::dispatch_Unique_easy(params.n_blocks,
                            stream,
                            pipe->u_morton_keys.data(),
                            pipe->u_unique_morton_keys.data(),
                            pipe->unique_tmp.u_flag_heads.data(),
                            params.n,
                            num_unique);

  SYNC_STREAM(stream);

  spdlog::info("num_unique: {}/{}", num_unique, params.n);
  pipe->n_unique_keys = num_unique;
  pipe->n_brt_nodes = num_unique - 1;

  gpu::dispatch_BuildRadixTree(params.n_blocks,
                               stream,
                               pipe->u_unique_morton_keys.data(),
                               pipe->brt.u_prefix_n.data(),
                               pipe->brt.u_has_leaf_left,
                               pipe->brt.u_has_leaf_right,
                               pipe->brt.u_left_child.data(),
                               pipe->brt.u_parent.data(),
                               pipe->n_unique_keys);

  gpu::dispatch_EdgeCount(params.n_blocks,
                          stream,
                          pipe->brt.u_prefix_n.data(),
                          pipe->brt.u_parent.data(),
                          pipe->u_edge_count.data(),
                          pipe->n_brt_nodes);

  gpu::dispatch_PrefixSum(params.n_blocks,
                          stream,
                          pipe->u_edge_count.data(),
                          pipe->u_edge_offset.data(),
                          pipe->prefix_sum_tmp.u_auxiliary.data(),
                          pipe->n_brt_nodes);
  SYNC_STREAM(stream);

  const auto n_oct_nodes = pipe->u_edge_offset[pipe->n_brt_nodes - 1];
  pipe->n_oct_nodes = n_oct_nodes;

  spdlog::info(
      "n_oct_nodes: {} ({}%)", n_oct_nodes, n_oct_nodes * 100.0f / params.n);

  gpu::dispatch_BuildOctree(
      params.n_blocks,
      stream,
      // --- output parameters ---
      pipe->oct.u_children,
      pipe->oct.u_corner,
      pipe->oct.u_cell_size,
      pipe->oct.u_child_node_mask,
      pipe->oct.u_child_leaf_mask,
      // --- end output parameters, begin input parameters (read-only)
      pipe->u_edge_offset.data(),
      pipe->u_edge_count.data(),
      pipe->u_unique_morton_keys.data(),
      pipe->brt.u_prefix_n.data(),
      pipe->brt.u_has_leaf_left,
      pipe->brt.u_has_leaf_right,
      pipe->brt.u_left_child.data(),
      pipe->brt.u_parent.data(),
      params.min_coord,
      params.getRange(),
      pipe->n_brt_nodes);

  SYNC_STREAM(stream);
}

void run_all_in_cpu(NaivePipe* pipe, const AppParams& params) {
  cpu::k_InitRandomVec4(pipe->u_points.data(),
                        params.n,
                        params.min_coord,
                        params.getRange(),
                        params.seed);

  cpu::k_ComputeMortonCode(pipe->u_points.data(),
                           pipe->u_morton_keys.data(),
                           params.n,
                           params.min_coord,
                           params.getRange());

  cpu::std_sort(pipe->u_morton_keys.data(), params.n);

  auto num_unique = cpu::std_unique(
      pipe->u_morton_keys.data(), pipe->u_unique_morton_keys.data(), params.n);
  pipe->n_unique_keys = num_unique;
  pipe->n_brt_nodes = num_unique - 1;

  spdlog::info("num_unique: {}/{}", num_unique, params.n);

  cpu::k_BuildRadixTree(pipe->n_unique_keys,
                        pipe->u_unique_morton_keys.data(),
                        pipe->brt.u_prefix_n.data(),
                        pipe->brt.u_has_leaf_left,
                        pipe->brt.u_has_leaf_right,
                        pipe->brt.u_left_child.data(),
                        pipe->brt.u_parent.data());

  cpu::k_EdgeCount(pipe->brt.u_prefix_n.data(),
                   pipe->brt.u_parent.data(),
                   pipe->u_edge_count.data(),
                   pipe->n_brt_nodes);

  cpu::std_exclusive_scan(
      pipe->u_edge_count.data(), pipe->u_edge_offset.data(), pipe->n_brt_nodes);
  const auto n_oct_nodes = pipe->u_edge_offset[pipe->n_brt_nodes - 1];
  pipe->n_oct_nodes = n_oct_nodes;

  spdlog::info(
      "n_oct_nodes: {} ({}%)", n_oct_nodes, n_oct_nodes * 100.0f / params.n);

  cpu::k_MakeOctNodes(pipe->oct.u_children,
                      pipe->oct.u_corner,
                      pipe->oct.u_cell_size,
                      pipe->oct.u_child_node_mask,
                      pipe->u_edge_offset.data(),
                      pipe->u_edge_count.data(),
                      pipe->u_unique_morton_keys.data(),
                      pipe->brt.u_prefix_n.data(),
                      pipe->brt.u_parent.data(),
                      params.min_coord,
                      params.getRange(),
                      pipe->n_brt_nodes);

  cpu::k_LinkLeafNodes(pipe->oct.u_children,
                       pipe->oct.u_child_leaf_mask,
                       pipe->u_edge_offset.data(),
                       pipe->u_edge_count.data(),
                       pipe->u_unique_morton_keys.data(),
                       pipe->brt.u_has_leaf_left,
                       pipe->brt.u_has_leaf_right,
                       pipe->brt.u_prefix_n.data(),
                       pipe->brt.u_parent.data(),
                       pipe->brt.u_left_child.data(),
                       pipe->n_brt_nodes);
}

int main(const int argc, const char** argv) {
  AppParams params(argc, argv);
  params.print_params();

  spdlog::set_level(params.debug_print ? spdlog::level::debug
                                       : spdlog::level::info);

  omp_set_num_threads(params.n_threads);
  if (params.debug_print) {
#pragma omp parallel
    { spdlog::info("Hello from thread {}", omp_get_thread_num()); }
  }

  // ------------------------------
  constexpr auto n_streams = 2;

  std::array<cudaStream_t, n_streams> streams;
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  }

  const auto pipe = std::make_unique<NaivePipe>(params.n);
  pipe->attachStream(streams[0]);

  if (params.use_cpu) {
    run_all_in_cpu(pipe.get(), params);
  } else {
    run_all_in_gpu(pipe.get(), params, streams[0]);
  }

  // ------------------------------

  spdlog::info("Done");
  for (const auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  }
  return 0;
}
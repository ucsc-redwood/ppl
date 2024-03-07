#include <omp.h>
#include <spdlog/spdlog.h>

#include "app_params.hpp"
#include "dispatcher.cuh"
#include "handlers/pipe.cuh"
#include "kernels_fwd.h"

void runAllStagesOnGpu(const AppParams& params,
                       const cudaStream_t stream,
                       const std::unique_ptr<Pipe>& pipe) {
  // CPU should handle input because this will be similar to the real
  // application. e.g., reading point cloud data from camera
  cpu::k_InitRandomVec4(
      pipe->u_points, pipe->n, pipe->min_coord, pipe->range, pipe->seed);

  gpu::v2::dispatch_ComputeMorton(params.n_blocks, stream, *pipe);
  gpu::v2::dispatch_RadixSort(params.n_blocks, stream, pipe->sort);
  gpu::v2::dispatch_RemoveDuplicates(
      params.n_blocks, stream, pipe->sort.data(), pipe->unique);

  SYNC_STREAM(stream);
  const auto n_unique = pipe->unique.attemptGetNumUnique();
  pipe->brt.setNumBrtNodes(n_unique - 1);

  gpu::v2::dispatch_BuildRadixTree(
      params.n_blocks, stream, pipe->unique.begin(), n_unique, pipe->brt);
  gpu::v2::dispatch_EdgeCount(
      params.n_blocks, stream, pipe->brt, pipe->u_edge_count);
  gpu::v2::dispatch_EdgeOffset_safe(params.n_blocks,
                                    stream,
                                    pipe->u_edge_count,
                                    pipe->u_edge_offset,
                                    pipe->brt.getNumBrtNodes());

  SYNC_STREAM(stream);
  // const auto n_unique = pipe->attemptGetNumOctNodes();
  // const auto n_oct_nodes = pipe->u_edge_offset[pipe->brt.getNumBrtNodes() -
  // 1];

  gpu::v2::dispatch_BuildOctree(params.n_blocks, stream, *pipe);

  // gpu::v2::dispatch_BuildOctree(params.n_blocks,
  //                               stream,
  //                               pipe->brt,
  //                               pipe->sort.data(),
  //                               pipe->u_edge_offset,
  //                               pipe->u_edge_count,
  //                               pipe->oct,
  //                               params.min_coord,
  //                               params.getRange());

  // Temporary disable this kernel on Windows, as it's not working on Windows,
  // but it's working on Linux

  // gpu::v2::dispatch_LinkOctreeNodes(params.n_blocks,
  //                                   stream,
  //                                   pipe->u_edge_offset,
  //                                   pipe->u_edge_count,
  //                                   pipe->sort.data(),
  //                                   pipe->brt,
  //                                   pipe->oct);

  SYNC_STREAM(stream);

  // // peek 10 oct nodes
  // for (auto i = 0; i < 10; ++i) {
  //   spdlog::trace("oct node[{}]: {}", i, pipe->oct.u_children[i][0]);
  // }

  // spdlog::info("Unique keys: {}/{} ({}%)",
  //              n_unique,
  //              pipe->n,
  //              100.0 * n_unique / pipe->n);
  // spdlog::info("Oct nodes: {}/{} ({}%)",
  //              n_oct_nodes,
  //              pipe->n,
  //              100.0 * n_oct_nodes / pipe->n);

  // merge the two spdlog calls, then set precision to 2 decimal places

  const auto n_oct_nodes = pipe->u_edge_offset[pipe->brt.getNumBrtNodes() - 1];

  spdlog::info("Unique keys: {} / {} ({}%) | Oct nodes: {} / {} ({}%)",
               n_unique,
               pipe->n,
               100.0f * n_unique / pipe->n,
               n_oct_nodes,
               pipe->n,
               100.0f * n_oct_nodes / pipe->n);
}

// todo : fix this later
unsigned int* temp_sort_alt;

void runAllStagesOnCpu(const AppParams& params,
                       const std::unique_ptr<Pipe>& pipe) {
  cpu::k_InitRandomVec4(
      pipe->u_points, pipe->n, pipe->min_coord, pipe->range, pipe->seed);

  cpu::k_ComputeMortonCode(params.n_threads,
                           pipe->u_points,
                           pipe->sort.u_sort,
                           pipe->n,
                           pipe->min_coord,
                           pipe->range);
  cpu::k_Sort(params.n_threads, pipe->sort.u_sort, temp_sort_alt, pipe->n);

  // There's no parrallel find duplicates on CPU, so just use std::unique for
  // now
  const auto it = std::unique(pipe->sort.u_sort, pipe->sort.end());
  const auto n_unique = std::distance(pipe->sort.u_sort, it);
  pipe->brt.setNumBrtNodes(n_unique - 1);

  // for cpu, u_sort is already sorted, and also removed dups
  cpu::k_BuildRadixTree(params.n_threads,
                        n_unique,
                        pipe->sort.u_sort,
                        pipe->brt.u_prefix_n,
                        pipe->brt.u_has_leaf_left,
                        pipe->brt.u_has_leaf_right,
                        pipe->brt.u_left_child,
                        pipe->brt.u_parent);

  cpu::k_EdgeCount(params.n_threads,
                   pipe->brt.u_prefix_n,
                   pipe->brt.u_parent,
                   pipe->u_edge_count,
                   pipe->brt.getNumBrtNodes());

  std::exclusive_scan(pipe->u_edge_count,
                      pipe->u_edge_count + pipe->brt.getNumBrtNodes(),
                      pipe->u_edge_offset,
                      0);

  // peek 10 edge offsets
  const auto n_oct_nodes = pipe->u_edge_offset[pipe->brt.getNumBrtNodes() - 1];

  // for (auto i = 0; i < 10; ++i) {
  //   spdlog::trace("edge offset[{}]: {}", i, pipe->u_edge_offset[i]);
  // }

  cpu::k_MakeOctNodes(params.n_threads,
                      pipe->oct.u_children,
                      pipe->oct.u_corner,
                      pipe->oct.u_cell_size,
                      pipe->oct.u_child_node_mask,
                      pipe->u_edge_offset,
                      pipe->u_edge_count,
                      pipe->sort.u_sort,
                      pipe->brt.u_prefix_n,
                      pipe->brt.u_parent,
                      pipe->min_coord,
                      pipe->range,
                      pipe->brt.getNumBrtNodes());

  cpu::k_LinkLeafNodes(params.n_threads,
                       pipe->oct.u_children,
                       pipe->oct.u_child_leaf_mask,
                       pipe->u_edge_offset,
                       pipe->u_edge_count,
                       pipe->sort.u_sort,
                       pipe->brt.u_has_leaf_left,
                       pipe->brt.u_has_leaf_right,
                       pipe->brt.u_prefix_n,
                       pipe->brt.u_parent,
                       pipe->brt.u_left_child,
                       pipe->brt.getNumBrtNodes());

  // peek 10 octree node u_children
  for (auto i = 0; i < 10; ++i) {
    spdlog::trace("oct node[{}]: {}", i, pipe->oct.u_children[i][0]);
  }

  spdlog::info("Unique keys: {} / {} ({}%) | Oct nodes: {} / {} ({}%)",
               n_unique,
               pipe->n,
               100.0f * n_unique / pipe->n,
               n_oct_nodes,
               pipe->n,
               100.0f * n_oct_nodes / pipe->n);

  // const auto is_sorted = std::is_sorted(pipe->sort.begin(),
  // pipe->sort.end()); spdlog::info("Is sorted: {}", is_sorted);
}

int main(const int argc, const char** argv) {
  AppParams params(argc, argv);
  params.print_params();

  temp_sort_alt = new unsigned int[params.n];

  switch (params.log_level) {
    case 0:
      spdlog::set_level(spdlog::level::off);
      break;
    case 1:
      spdlog::set_level(spdlog::level::info);
      break;
    case 2:
      spdlog::set_level(spdlog::level::debug);
      break;
    case 3:
      spdlog::set_level(spdlog::level::trace);
      break;
    default:
      spdlog::set_level(spdlog::level::info);
      break;
  }

  omp_set_num_threads(params.n_threads);
#pragma omp parallel
  { spdlog::debug("Hello from thread {}", omp_get_thread_num()); }

  // ------------------------------
  constexpr auto n_streams = 1;
  const auto n_iterations = params.n_iterations;

  std::array<cudaStream_t, n_streams> streams;
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  }

  const auto pipe = std::make_unique<Pipe>(
      params.n, params.min_coord, params.getRange(), params.seed);
  pipe->attachStreamGlobal(streams[0]);

  if (params.use_cpu) {
    const auto start = std::chrono::high_resolution_clock::now();

    for (auto i = 0; i < n_iterations; ++i) {
      ++pipe->seed;
      runAllStagesOnCpu(params, pipe);
    }

    const auto end = std::chrono::high_resolution_clock::now();

    spdlog::set_level(spdlog::level::info);

    // print in milliseconds
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    spdlog::info("Total time: {} ms | Average time: {} ms",
                 elapsed.count(),
                 elapsed.count() / n_iterations);

  } else {
    cudaEvent_t start, stop;
    CHECK_CUDA_CALL(cudaEventCreate(&start));
    CHECK_CUDA_CALL(cudaEventCreate(&stop));

    CHECK_CUDA_CALL(cudaEventRecord(start, 0));

    for (auto i = 0; i < n_iterations; ++i) {
      ++pipe->seed;
      runAllStagesOnGpu(params, streams[0], pipe);
    }

    CHECK_CUDA_CALL(cudaEventRecord(stop, 0));
    CHECK_CUDA_CALL(cudaEventSynchronize(stop));

    spdlog::set_level(spdlog::level::info);

    float milliseconds = 0;
    CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    spdlog::info("Total time: {} ms | Average time: {} ms",
                 milliseconds,
                 milliseconds / n_iterations);
  }
  // ------------------------------

  spdlog::info("Done");
  for (const auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  }
  return 0;
}
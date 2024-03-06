#include <omp.h>
#include <spdlog/spdlog.h>

#include "app_params.hpp"
#include "handlers/pipe.cuh"
#include "kernels_fwd.h"

void runAllStagesOnGpu(const AppParams& params,
                       const cudaStream_t stream,
                       const std::unique_ptr<Pipe>& pipe) {
  gpu::v2::dispatch_Init(params.n_blocks, stream, *pipe);
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
  const auto n_oct_nodes = pipe->u_edge_offset[pipe->brt.getNumBrtNodes() - 1];

  gpu::v2::dispatch_BuildOctree(params.n_blocks,
                                stream,
                                pipe->brt,
                                pipe->sort.data(),
                                pipe->u_edge_offset,
                                pipe->u_edge_count,
                                pipe->oct,
                                params.min_coord,
                                params.getRange());

  gpu::v2::dispatch_LinkOctreeNodes(params.n_blocks,
                                    stream,
                                    pipe->u_edge_offset,
                                    pipe->u_edge_count,
                                    pipe->sort.data(),
                                    pipe->brt,
                                    pipe->oct);

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

  spdlog::info("Unique keys: {} / {} ({}%) | Oct nodes: {} / {} ({}%)",
               n_unique,
               pipe->n,
               100.0f * n_unique / pipe->n,
               n_oct_nodes,
               pipe->n,
               100.0f * n_oct_nodes / pipe->n);
}

int main(const int argc, const char** argv) {
  AppParams params(argc, argv);
  params.print_params();

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

  for (auto i = 0; i < n_iterations; ++i) {
    ++pipe->seed;
    runAllStagesOnGpu(params, streams[0], pipe);
  }

  // ------------------------------

  spdlog::info("Done");
  for (const auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  }
  return 0;
}
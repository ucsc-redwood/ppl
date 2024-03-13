#include <cuda_runtime_api.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "../app_params.hpp"
#include "cuda/helper.cuh"
#include "device_dispatcher.h"
#include "handlers/pipe.h"
#include "host_dispatcher.h"
#include "openmp/kernels/00_init.hpp"

[[maybe_unused]] void attachPipeToStream(const cudaStream_t stream,
                                         const Pipe* p) {
  // Pipe
  ATTACH_STREAM_SINGLE(p->u_points);
  ATTACH_STREAM_SINGLE(p->u_edge_count);
  ATTACH_STREAM_SINGLE(p->u_edge_offset);

  // One sweep
  ATTACH_STREAM_SINGLE(p->sort.u_sort);
  ATTACH_STREAM_SINGLE(p->sort.u_sort_alt);

  // Unique
  ATTACH_STREAM_SINGLE(p->unique.u_keys_out);

  // Radix tree
  ATTACH_STREAM_SINGLE(p->brt.u_prefix_n);
  ATTACH_STREAM_SINGLE(p->brt.u_has_leaf_left);
  ATTACH_STREAM_SINGLE(p->brt.u_has_leaf_right);
  ATTACH_STREAM_SINGLE(p->brt.u_left_child);
  ATTACH_STREAM_SINGLE(p->brt.u_parent);

  // Octree
  ATTACH_STREAM_SINGLE(p->oct.u_children);
  ATTACH_STREAM_SINGLE(p->oct.u_corner);
  ATTACH_STREAM_SINGLE(p->oct.u_cell_size);
  ATTACH_STREAM_SINGLE(p->oct.u_child_node_mask);
  ATTACH_STREAM_SINGLE(p->oct.u_child_leaf_mask);

  SYNC_STREAM(stream);
}

// ----------------------------------------------------------------
// Baselines
// ----------------------------------------------------------------

// 1.
void runAllStagesOnGpu(const AppParams& params,
                       const cudaStream_t stream,
                       const std::unique_ptr<Pipe>& pipe) {
  pipe->acquireNextFrameData();

  gpu::v2::dispatch_ComputeMorton(params.n_blocks, stream, *pipe);
  gpu::v2::dispatch_RadixSort(params.n_blocks, stream, *pipe);
  gpu::v2::dispatch_RemoveDuplicates(params.n_blocks, stream, *pipe);
  gpu::v2::dispatch_BuildRadixTree(params.n_blocks, stream, *pipe);
  gpu::v2::dispatch_EdgeCount(params.n_blocks, stream, *pipe);
  gpu::v2::dispatch_EdgeOffset(params.n_blocks, stream, *pipe);
  gpu::v2::dispatch_BuildOctree(params.n_blocks, stream, *pipe);

  SYNC_STREAM(stream);

  spdlog::info("Unique keys: {} / {} ({}%) | Oct nodes: {} / {} ({}%)",
               pipe->getUniqueSize(),
               pipe->n,
               100.0f * pipe->getUniqueSize() / pipe->n,
               pipe->getOctSize(),
               pipe->n,
               100.0f * pipe->getOctSize() / pipe->n);
}

// 2.
void runBestOnEachPU(const AppParams& params,
                     const cudaStream_t stream,
                     const std::unique_ptr<Pipe>& pipe) {
  pipe->acquireNextFrameData();

  gpu::v2::dispatch_ComputeMorton(params.n_blocks, stream, *pipe);
  gpu::v2::dispatch_RadixSort(params.n_blocks, stream, *pipe);
  SYNC_STREAM(stream);
  cpu::v2::dispatch_RemoveDuplicates(params.n_threads, *pipe);
  cpu::v2::dispatch_BuildRadixTree(params.n_threads, *pipe);
  gpu::v2::dispatch_EdgeCount(params.n_blocks, stream, *pipe);
  SYNC_STREAM(stream);
  cpu::v2::dispatch_EdgeOffset(params.n_threads, *pipe);
  gpu::v2::dispatch_BuildOctree(params.n_blocks, stream, *pipe);
  SYNC_STREAM(stream);

  spdlog::info("Unique keys: {} / {} ({}%) | Oct nodes: {} / {} ({}%)",
               pipe->getUniqueSize(),
               pipe->n,
               100.0f * pipe->getUniqueSize() / pipe->n,
               pipe->getOctSize(),
               pipe->n,
               100.0f * pipe->getOctSize() / pipe->n);
}

// ----------------------------------------------------------------
// Method 1: Two-Phase Coarse-Grained
//
// ----------------------------------------------------------------

// 3.
void runTwoPhaseCorseGrained(const AppParams& params,
                             const cudaStream_t stream,
                             const std::unique_ptr<Pipe>& pipe) {}

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
  // Init
  // ------------------------------

  constexpr auto n_streams = 2;
  const auto n_iterations = params.n_iterations;

  std::array<cudaStream_t, n_streams> streams;
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  }

  const auto pipe = std::make_unique<Pipe>(
      params.n, params.min_coord, params.getRange(), params.seed);
  cpu::k_InitRandomVec4(
      pipe->u_points, pipe->n, pipe->min_coord, pipe->range, pipe->seed);

  // attachPipeToStream(streams[0], pipe.get());

  // ------------------------------

  cudaEvent_t start, stop;
  CHECK_CUDA_CALL(cudaEventCreate(&start));
  CHECK_CUDA_CALL(cudaEventCreate(&stop));

  CHECK_CUDA_CALL(cudaEventRecord(start, nullptr));

  switch (params.method) {
    case 0:
      for (auto i = 0; i < n_iterations; ++i) {
        runAllStagesOnGpu(params, streams[0], pipe);
      }
      break;
    case 1:
      for (auto i = 0; i < n_iterations; ++i) {
        runBestOnEachPU(params, streams[0], pipe);
      }
      break;
    case 2:
      for (auto i = 0; i < n_iterations; ++i) {
        runTwoPhaseCorseGrained(params, streams[0], pipe);
      }
      break;
    default:
      spdlog::error("Unknown method: {}", params.method);
      throw std::runtime_error("Unknown method");
  }

  CHECK_CUDA_CALL(cudaEventRecord(stop, nullptr));
  CHECK_CUDA_CALL(cudaEventSynchronize(stop));

  spdlog::set_level(spdlog::level::info);

  float milliseconds = 0;
  CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  spdlog::info("Total time: {} ms | Average time: {} ms",
               milliseconds,
               milliseconds / n_iterations);
  // ------------------------------

  spdlog::info("Done");
  for (const auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  }
  return 0;
}
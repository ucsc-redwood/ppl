#include <omp.h>
#include <spdlog/spdlog.h>

#include "../app_params.hpp"
#include "cuda/helper.cuh"
#include "device_dispatcher.h"
#include "handlers/pipe.h"
#include "host_dispatcher.h"
#include "openmp/kernels/00_init.hpp"

[[maybe_unused]] void attachPipeToStream(const cudaStream_t stream, Pipe* p) {
  // Pipe
  ATTACH_STREAM_GLOBAL(p->u_points);
  ATTACH_STREAM_GLOBAL(p->u_edge_count);
  ATTACH_STREAM_GLOBAL(p->u_edge_offset);

  // One sweep
  ATTACH_STREAM_GLOBAL(p->sort.u_sort);
  ATTACH_STREAM_GLOBAL(p->sort.u_sort_alt);

  // Unique
  ATTACH_STREAM_GLOBAL(p->unique.u_keys_out);

  // Radix tree
  ATTACH_STREAM_GLOBAL(p->brt.u_prefix_n);
  ATTACH_STREAM_GLOBAL(p->brt.u_has_leaf_left);
  ATTACH_STREAM_GLOBAL(p->brt.u_has_leaf_right);
  ATTACH_STREAM_GLOBAL(p->brt.u_left_child);
  ATTACH_STREAM_GLOBAL(p->brt.u_parent);

  // Octree
  ATTACH_STREAM_GLOBAL(p->oct.u_children);
  ATTACH_STREAM_GLOBAL(p->oct.u_corner);
  ATTACH_STREAM_GLOBAL(p->oct.u_cell_size);
  ATTACH_STREAM_GLOBAL(p->oct.u_child_node_mask);
  ATTACH_STREAM_GLOBAL(p->oct.u_child_leaf_mask);

  SYNC_STREAM(stream);
}

void runAllStagesOnGpu(const AppParams& params,
                       const cudaStream_t stream,
                       const std::unique_ptr<Pipe>& pipe) {
  // CPU should handle input because this will be similar to the real
  // application. e.g., reading point cloud data from camera

  cpu::k_InitRandomVec4(
      pipe->u_points, pipe->n, pipe->min_coord, pipe->range, pipe->seed);

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

void runAllStagesOnCpu(const AppParams& params,
                       const std::unique_ptr<Pipe>& pipe) {
  cpu::k_InitRandomVec4(
      pipe->u_points, pipe->n, pipe->min_coord, pipe->range, pipe->seed);

  cpu::v2::dispatch_ComputeMorton(params.n_threads, *pipe);
  cpu::v2::dispatch_RadixSort(params.n_threads, *pipe);
  cpu::v2::dispatch_RemoveDuplicates(params.n_threads, *pipe);
  cpu::v2::dispatch_BuildRadixTree(params.n_threads, *pipe);
  cpu::v2::dispatch_EdgeCount(params.n_threads, *pipe);
  cpu::v2::dispatch_EdgeOffset(params.n_threads, *pipe);
  cpu::v2::dispatch_BuildOctree(params.n_threads, *pipe);

  spdlog::info("Unique keys: {} / {} ({}%) | Oct nodes: {} / {} ({}%)",
               pipe->getUniqueSize(),
               pipe->n,
               100.0f * pipe->getUniqueSize() / pipe->n,
               pipe->getOctSize(),
               pipe->n,
               100.0f * pipe->getOctSize() / pipe->n);
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
  constexpr auto n_streams = 2;
  const auto n_iterations = params.n_iterations;

  std::array<cudaStream_t, n_streams> streams;
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  }

  const auto pipe = std::make_unique<Pipe>(
      params.n, params.min_coord, params.getRange(), params.seed);
  // attachPipeToStream(streams[0], pipe.get());

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

    CHECK_CUDA_CALL(cudaEventRecord(start, nullptr));

    for (auto i = 0; i < n_iterations; ++i) {
      ++pipe->seed;
      runAllStagesOnGpu(params, streams[0], pipe);
    }

    CHECK_CUDA_CALL(cudaEventRecord(stop, nullptr));
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
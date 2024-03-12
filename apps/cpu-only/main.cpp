#include <omp.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <memory>

#include "../app_params.hpp"
#include "handlers/pipe.h"
#include "host_dispatcher.h"
#include "openmp/kernels/00_init.hpp"

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

void runAllStagesOnCpu(const AppParams& params,
                       const std::unique_ptr<Pipe>& pipe) {
  pipe->acquireNextFrameData();

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


int main(const int argc, const char* argv[]) {
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

  // ----------------------------------------------------------------
  // Initialization
  // ----------------------------------------------------------------

  const auto n_iterations = params.n_iterations;

  const auto pipe = std::make_unique<Pipe>(
      params.n, params.min_coord, params.getRange(), params.seed);
  cpu::k_InitRandomVec4(
      pipe->u_points, pipe->n, pipe->min_coord, pipe->range, pipe->seed);

  const auto start = std::chrono::high_resolution_clock::now();

  for (auto i = 0; i < n_iterations; ++i) {
    ++pipe->seed;
    runAllStagesOnCpu(params, pipe);
  }

  const auto end = std::chrono::high_resolution_clock::now();

  // ----------------------------------------------------------------

  spdlog::set_level(spdlog::level::info);

  // print in milliseconds
  const std::chrono::duration<double, std::milli> elapsed = end - start;
  spdlog::info("Total time: {} ms | Average time: {} ms",
               elapsed.count(),
               elapsed.count() / n_iterations);

  return 0;
}
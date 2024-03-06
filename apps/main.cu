#include <omp.h>
#include <spdlog/spdlog.h>

#include "app_params.hpp"
#include "handlers/pipe.cuh"
#include "kernels_fwd.h"

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

  for (auto i = 0; i < n_iterations; ++i) {
    gpu::v2::dispatch_Init(params.n_blocks, streams[0], *pipe);

    gpu::v2::dispatch_ComputeMorton(params.n_blocks, streams[0], *pipe);

    gpu::v2::dispatch_RadixSort(params.n_blocks, streams[0], pipe->sort);

    SYNC_STREAM(streams[0]);

    auto is_sorted = std::is_sorted(pipe->sort.begin(), pipe->sort.end());
    spdlog::info("Is sorted (after): {}", is_sorted);
  }

  // ------------------------------

  spdlog::info("Done");
  for (const auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  }
  return 0;
}
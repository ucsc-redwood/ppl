#include <spdlog/spdlog.h>

// #include <algorithm>
#include <iostream>

#include "app_params.hpp"
// #include "dispatch_kernels.cuh"
#include "types/pipe.cuh"

int main(const int argc, const char** argv) {
  AppParams params(argc, argv);
  params.print_params();

  spdlog::set_level(spdlog::level::debug);

  // // ------------------------------
  // constexpr auto n_streams = 1;

  // std::array<cudaStream_t, n_streams> streams;
  // for (auto stream : streams) {
  //   CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  // }

  // const std::array<std::unique_ptr<Pipe>, n_streams> pipes{
  //     std::make_unique<Pipe>(params.n),
  // };

  // for (auto i = 0; i < n_streams; ++i) {
  //   pipes[i]->attachStream(streams[i]);
  // }

  // {
  //   gpu::dispatchKernel(gpu::k_InitRandomVec4,
  //                       params.n_blocks,
  //                       streams[0],
  //                       // *** original arguments start ***
  //                       pipes[0]->getPoints(),
  //                       params.n,
  //                       params.min_coord,
  //                       params.getRange(),
  //                       params.seed);

  //   gpu::dispatchKernel(gpu::k_ComputeMortonCode,
  //                       params.n_blocks,
  //                       streams[0],
  //                       // *** original arguments start ***
  //                       pipes[0]->getPoints(),
  //                       pipes[0]->getMortonKeys(),
  //                       params.n,
  //                       params.min_coord,
  //                       params.getRange());

  //   gpu::dispatch_RadixSort(params.n_blocks, streams[0], pipes[0]->onesweep);

  //   SYNC_STREAM(streams[0]);
  // }

  // // ------------------------------

  // auto is_sorted = std::is_sorted(pipes[0]->onesweep.getSort(),
  //                                 pipes[0]->onesweep.getSort() + params.n);
  // spdlog::info("is_sorted: {}", is_sorted);

  spdlog::info("Done");
  // for (const auto stream : streams) {
  //   CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  // }
  return 0;
}
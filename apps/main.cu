#include <spdlog/spdlog.h>

#include <algorithm>
#include <iostream>

#include "app_params.hpp"
#include "dispatch_kernels.cuh"
#include "types/pipe.cuh"

int main(const int argc, const char** argv) {
  AppParams params;
  params.n = 1920 * 1080;  // 2.0736M
  params.min_coord = 0.0f;
  params.max_coord = 1.0f;
  params.seed = 114514;
  params.n_threads = 4;
  params.n_blocks = 16;

  // set spdlog level
  spdlog::set_level(spdlog::level::debug);

  // ------------------------------
  constexpr auto n_streams = 1;

  std::array<cudaStream_t, n_streams> streams;
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  }

  std::array<std::unique_ptr<Pipe>, n_streams> pipes{
      std::make_unique<Pipe>(params.n),
  };

  for (auto i = 0; i < n_streams; ++i) {
    pipes[i]->attachStream(streams[i]);
  }

  {
    gpu::dispatchKernel(gpu::k_InitRandomVec4,
                        params.n_blocks,
                        streams[0],
                        // *** original arguments start ***
                        pipes[0]->getPoints(),
                        params.n,
                        params.min_coord,
                        params.getRange(),
                        params.seed);

    gpu::dispatchKernel(gpu::k_ComputeMortonCode,
                        params.n_blocks,
                        streams[0],
                        // *** original arguments start ***
                        pipes[0]->getPoints(),
                        pipes[0]->getMortonKeys(),
                        params.n,
                        params.min_coord,
                        params.getRange());

    gpu::dispatch_RadixSort(params.n_blocks, streams[0], pipes[0]->onesweep);

    gpu::dispatch_RemoveDuplicates(params.n_blocks,
                                   streams[0],
                                   pipes[0]->getMortonKeys(),
                                   pipes[0]->unique);

    SYNC_STREAM(streams[0]);
  }

  // ------------------------------

  // // peek all unique points
  // for (auto i = 0; i < 32; ++i) {
  //   spdlog::info("unique[{}]: {}", i, pipes[0]->unique.u_unique_keys[i]);
  // }

  auto is_sorted = std::is_sorted(pipes[0]->onesweep.getSort(),
                                  pipes[0]->onesweep.getSort() + params.n);
  spdlog::info("is_sorted: {}", is_sorted);

  // // peek u sort
  // for (auto i = 0; i < 10; ++i) {
  //   spdlog::info("u_sort[{}]: {}", i, pipes[0]->onesweep.getSort()[i]);
  // }

  spdlog::info("Done");
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  }
  return 0;
}
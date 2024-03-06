#pragma once

#include "benchmark_helper.cuh"
#include "config.h"

static void BM_Morton(bm::State& st) {
  constexpr auto config = configs[0];
  const auto grid_size = st.range(0);
  const auto block_size =
      DetermineBlockSizeAndDisplay(gpu::k_ComputeMortonCode, st);

  glm::vec4* u_points;
  unsigned int* u_sort;
  MALLOC_MANAGED(&u_points, config.n);
  MALLOC_MANAGED(&u_sort, config.n);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);
    gpu::k_ComputeMortonCode<<<grid_size, block_size>>>(
        u_points, u_sort, config.n, config.min_coord, config.range);
  }

  CUDA_FREE(u_points);
  CUDA_FREE(u_sort);
}

BENCHMARK(BM_Morton)
    ->UseManualTime()
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->ArgName("GridSize");

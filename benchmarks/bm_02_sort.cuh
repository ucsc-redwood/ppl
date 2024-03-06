#pragma once

#include "config.h"
#include "cu_bench_helper.cuh"
#include "handlers/one_sweep.cuh"

static void BM_Sort(bm::State& st) {
  CREATE_STREAM;

  constexpr auto config = configs[0];
  const auto grid_size = st.range(0);

  OneSweepHandler handler(config.n);

  std::generate(handler.begin(), handler.end(), []() { return rand(); });

  handler.attachStreamGlobal(stream);
  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::v2::dispatch_RadixSort(grid_size, 0, handler);
  }

  DESCROY_STREAM;
}

BENCHMARK(BM_Sort)
    ->UseManualTime()
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 256)
    ->ArgName("GridSize");

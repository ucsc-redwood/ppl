#pragma once

#include "bm_02_sort.cuh"
#include "config.h"
#include "handlers/one_sweep.cuh"

void BM_GPU_Sort(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto grid_size = st.range(0);

  st.counters["hist_block_size"] = OneSweepHandler::GLOBAL_HIST_THREADS;
  st.counters["bin_block_size"] = OneSweepHandler::BINNING_THREADS;

  OneSweepHandler handler(n);

  std::generate(
      handler.begin(), handler.end(), [n = n]() mutable { return --n; });

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::k_GlobalHistogram<<<grid_size, OneSweepHandler::GLOBAL_HIST_THREADS>>>(
        handler.data(), handler.im_storage.d_global_histogram, n);

    gpu::k_Scan<<<OneSweepHandler::RADIX_PASSES, OneSweepHandler::RADIX>>>(
        handler.im_storage.d_global_histogram,
        handler.im_storage.d_first_pass_histogram,
        handler.im_storage.d_second_pass_histogram,
        handler.im_storage.d_third_pass_histogram,
        handler.im_storage.d_fourth_pass_histogram);

    gpu::k_DigitBinningPass<<<grid_size, OneSweepHandler::BINNING_THREADS>>>(
        handler.u_sort,
        handler.u_sort_alt,
        handler.im_storage.d_first_pass_histogram,
        handler.im_storage.d_index,
        n,
        0);

    gpu::k_DigitBinningPass<<<grid_size, OneSweepHandler::BINNING_THREADS>>>(
        handler.u_sort_alt,
        handler.u_sort,
        handler.im_storage.d_second_pass_histogram,
        handler.im_storage.d_index,
        n,
        8);

    gpu::k_DigitBinningPass<<<grid_size, OneSweepHandler::BINNING_THREADS>>>(
        handler.u_sort,
        handler.u_sort_alt,
        handler.im_storage.d_third_pass_histogram,
        handler.im_storage.d_index,
        n,
        16);

    gpu::k_DigitBinningPass<<<grid_size, OneSweepHandler::BINNING_THREADS>>>(
        handler.u_sort_alt,
        handler.u_sort,
        handler.im_storage.d_fourth_pass_histogram,
        handler.im_storage.d_index,
        n,
        24);
  }
}

void BM_CPU_Sort(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto n_threads = st.range(0);

  std::vector<unsigned int> h_sort(n);
  std::vector<unsigned int> h_sort_alt(n);

  std::generate(
      h_sort.begin(), h_sort.end(), [n = n]() mutable { return --n; });

  for (auto _ : st) {
    cpu::k_Sort(n_threads, h_sort.data(), h_sort_alt.data(), n);
  }
}

BENCHMARK(BM_GPU_Sort)
    ->UseManualTime()
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

BENCHMARK(BM_CPU_Sort)
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->ArgName("Threads");
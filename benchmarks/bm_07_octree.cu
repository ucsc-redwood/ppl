#include "bm_07_octree.cuh"

#include <dispatcher.h>

#include "config.h"
#include "cuda/kernels/00_init.cuh"
#include "handlers/pipe.cuh"

void BM_GPU_Octree(bm::State& st) {
  CREATE_STREAM

  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto grid_size = st.range(0);

  Pipe p(n, min_coord, range, init_seed);
  p.attachStreamGlobal(stream);

  cpu::k_InitRandomVec4(p.u_points, n, min_coord, range, init_seed);
  gpu::v2::dispatch_ComputeMorton(16, stream, p);
  gpu::v2::dispatch_RadixSort(16, stream, p);
  gpu::v2::dispatch_RemoveDuplicates(16, stream, p);
  gpu::v2::dispatch_BuildRadixTree(16, stream, p);
  gpu::v2::dispatch_EdgeCount(16, stream, p);
  gpu::v2::dispatch_EdgeOffset(16, stream, p);
  SYNC_STREAM(stream);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::v2::dispatch_BuildOctree(grid_size, stream, p);
  }

  DESCROY_STREAM
}

void BM_CPU_Octree(bm::State& st) {
  CREATE_STREAM
  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto n_threads = st.range(0);

  Pipe p(n, min_coord, range, init_seed);
  p.attachStreamGlobal(stream);

  cpu::k_InitRandomVec4(p.u_points, n, min_coord, range, init_seed);
  gpu::v2::dispatch_ComputeMorton(16, stream, p);
  gpu::v2::dispatch_RadixSort(16, stream, p);
  gpu::v2::dispatch_RemoveDuplicates(16, stream, p);
  gpu::v2::dispatch_BuildRadixTree(16, stream, p);
  gpu::v2::dispatch_EdgeCount(16, stream, p);
  gpu::v2::dispatch_EdgeOffset(16, stream, p);
  SYNC_STREAM(stream);

  for (auto _ : st) {
    cpu::v2::dispatch_BuildOctree(n_threads, p);
  }

  DESCROY_STREAM
}

BENCHMARK(BM_GPU_Octree)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize")
    ->Iterations(1000);

BENCHMARK(BM_CPU_Octree)
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->ArgName("Threads");

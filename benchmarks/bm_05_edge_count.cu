#include "bm_05_edge_count.cuh"

#include "config.h"
#include "handlers/radix_tree.cuh"

void BM_GPU_EdgeCount(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto grid_size = st.range(0);

  const auto block_size =
      determineBlockSizeAndDisplay(gpu::k_EdgeCount, st);

  unsigned int* u_morton;
  MALLOC_MANAGED(&u_morton, n);

  std::iota(u_morton, u_morton + n, 0);

  const RadixTree brt(n);

  int* u_edges;
  MALLOC_MANAGED(&u_edges, n);

  gpu::k_BuildRadixTree<<<16, 768>>>(n,
                                     u_morton,
                                     brt.u_prefix_n,
                                     brt.u_has_leaf_left,
                                     brt.u_has_leaf_right,
                                     brt.u_left_child,
                                     brt.u_parent);
  SYNC_DEVICE();

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::k_EdgeCount<<<grid_size, block_size>>>(
        brt.u_prefix_n,
        brt.u_parent,
        u_edges,
        n);
  }

  CUDA_FREE(u_morton);
  CUDA_FREE(u_edges);
}

void BM_CPU_EdgeCount(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto n_threads = st.range(0);

  unsigned int* u_morton;
  MALLOC_MANAGED(&u_morton, n);

  const RadixTree brt(n);

  std::vector<int> u_edges(n);

  gpu::k_BuildRadixTree<<<16, 768>>>(n,
                                     u_morton,
                                     brt.u_prefix_n,
                                     brt.u_has_leaf_left,
                                     brt.u_has_leaf_right,
                                     brt.u_left_child,
                                     brt.u_parent);
  SYNC_DEVICE();

  for (auto _ : st) {
    cpu::k_EdgeCount(n_threads,
                     brt.u_prefix_n,
                     brt.u_parent,
                     u_edges.data(),
                     n);
  }

  CUDA_FREE(u_morton);
}

BENCHMARK(BM_GPU_EdgeCount)
    ->UseManualTime()
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->Iterations(300) // takes too long
    ->ArgName("GridSize");

BENCHMARK(BM_CPU_EdgeCount)
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->Iterations(300) // takes too long
    ->ArgName("NumThreads");
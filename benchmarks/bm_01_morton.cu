#include "bm_01_morton.cuh"
#include "config.h"

void BM_GPU_Morton(bm::State& st) {
  const auto [n, min_coord, range, _] = configs[0];
  const auto grid_size = st.range(0);
  const auto block_size =
      determineBlockSizeAndDisplay(gpu::k_ComputeMortonCode, st);

  glm::vec4* u_points;
  unsigned int* u_sort;
  MALLOC_MANAGED(&u_points, n);
  MALLOC_MANAGED(&u_sort, n);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);
    gpu::k_ComputeMortonCode<<<grid_size, block_size>>>(
        u_points, u_sort, n, min_coord, range);
  }

  CUDA_FREE(u_points);
  CUDA_FREE(u_sort);
}

void BM_CPU_Morton(bm::State& st) {
  const auto [n, min_coord, range, _] = configs[0];
  const auto n_threads = static_cast<int>(st.range(0));

  std::vector<glm::vec4> h_points(n);
  std::vector<unsigned int> h_sort(n);

  for (auto _ : st) {
    cpu::k_ComputeMortonCode(
        n_threads, h_points.data(), h_sort.data(), n, min_coord, range);
  }
}

BENCHMARK(BM_GPU_Morton)
    ->UseManualTime()
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

BENCHMARK(BM_CPU_Morton)
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->ArgName("Threads");

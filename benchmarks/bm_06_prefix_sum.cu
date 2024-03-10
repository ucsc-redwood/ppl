#include "bm_06_prefix_sum.cuh"
#include "config.h"

void BM_GPU_PrefixSum(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];

  constexpr auto block_size = gpu::PrefixSumAgent<int>::n_threads;
  st.counters["block_size"] = block_size;

  int* u_data;
  int* u_data_out;
  MALLOC_MANAGED(&u_data, n);
  MALLOC_MANAGED(&u_data_out, n);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::k_SingleBlockExclusiveScan<<<1, block_size>>>(u_data, u_data_out, n);
  }

  CUDA_FREE(u_data);
  CUDA_FREE(u_data_out);
}

void BM_CPU_PrefixSum(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];

  std::vector u_data(n, 1);
  std::vector<int> u_data_out(n);

  for (auto _ : st) {
    std::partial_sum(u_data.begin(), u_data.end(), u_data_out.begin());
  }
}

BENCHMARK(BM_GPU_PrefixSum)->UseManualTime()->Unit(bm::kMillisecond);
BENCHMARK(BM_CPU_PrefixSum)->Unit(bm::kMillisecond);

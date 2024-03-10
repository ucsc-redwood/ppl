#include "bm_03_unique.cuh"
#include "config.h"

void BM_GPU_Unique(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto grid_size = st.range(0);

  constexpr auto unique_block_size = 256; // in 'agent.cu'
  constexpr auto prefix_block_size = 128;

  st.counters["unique_block_size"] = unique_block_size;
  st.counters["prefix_block_size"] = prefix_block_size;

  unsigned int* u_sort;
  int* u_flag_heads;
  unsigned int* u_keys_out;

  MALLOC_MANAGED(&u_sort, n);
  MALLOC_MANAGED(&u_flag_heads, n);
  MALLOC_MANAGED(&u_keys_out, n);
  SYNC_DEVICE();

  std::iota(u_sort, u_sort + n, 0);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::k_FindDups<<<grid_size, unique_block_size>>>(u_sort, u_flag_heads, n);

    gpu::k_SingleBlockExclusiveScan<<<1, prefix_block_size>>>(
        u_flag_heads,
        u_flag_heads,
        n);
    gpu::k_MoveDups<<<grid_size, unique_block_size>>>(
        u_sort,
        u_flag_heads,
        n,
        u_keys_out,
        nullptr);
  }

  CUDA_FREE(u_sort);
  CUDA_FREE(u_flag_heads);
  CUDA_FREE(u_keys_out);
}

void BM_CPU_Unique(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];

  std::vector<unsigned int> data(n);
  std::iota(data.begin(), data.end(), 0);

  for (auto _ : st) {
    const auto it = std::unique(data.begin(), data.end());
    const auto n_unique = std::distance(data.begin(), it);
  }
}

BENCHMARK(BM_GPU_Unique)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

BENCHMARK(BM_CPU_Unique)->Unit(bm::kMillisecond);
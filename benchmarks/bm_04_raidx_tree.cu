#include "bm_04_raidx_tree.cuh"
#include "config.h"

void BM_GPU_RadixTree(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto grid_size = st.range(0);

  const auto block_size =
      determineBlockSizeAndDisplay(gpu::k_BuildRadixTree, st);

  unsigned int* u_sorted_unique_morton;

  const auto n_unique = n;
  const auto n_brt_nodes = n_unique - 1;

  MALLOC_MANAGED(&u_sorted_unique_morton, n_brt_nodes);

  std::iota(u_sorted_unique_morton, u_sorted_unique_morton + n_brt_nodes, 0);

  uint8_t* d_prefix_n;
  bool* d_has_leaf_left;
  bool* d_has_leaf_right;
  int* d_left_child;
  int* d_parent;

  MALLOC_DEVICE(&d_prefix_n, n_brt_nodes);
  MALLOC_DEVICE(&d_has_leaf_left, n_brt_nodes);
  MALLOC_DEVICE(&d_has_leaf_right, n_brt_nodes);
  MALLOC_DEVICE(&d_left_child, n_brt_nodes);
  MALLOC_DEVICE(&d_parent, n_brt_nodes);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::k_BuildRadixTree<<<grid_size, block_size>>>(n_unique,
                                                     u_sorted_unique_morton,
                                                     d_prefix_n,
                                                     d_has_leaf_left,
                                                     d_has_leaf_right,
                                                     d_left_child,
                                                     d_parent);
  }

  CUDA_FREE(u_sorted_unique_morton);
  CUDA_FREE(d_prefix_n);
  CUDA_FREE(d_has_leaf_left);
  CUDA_FREE(d_has_leaf_right);
  CUDA_FREE(d_left_child);
  CUDA_FREE(d_parent);
}

void BM_CPU_RadixTree(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto n_thread = st.range(0);

  const auto n_unique = n;
  const auto n_brt_nodes = n_unique - 1;

  std::vector<unsigned int> h_sorted_unique_morton(n_brt_nodes);
  std::vector<uint8_t> h_prefix_n(n_brt_nodes);
  auto h_has_leaf_left = new bool[n_brt_nodes];
  auto h_has_leaf_right = new bool[n_brt_nodes];
  std::vector<int> h_left_child(n_brt_nodes);
  std::vector<int> h_parent(n_brt_nodes);

  for (auto _ : st) {
    cpu::k_BuildRadixTree(n_thread,
                          n_unique,
                          h_sorted_unique_morton.data(),
                          h_prefix_n.data(),
                          h_has_leaf_left,
                          h_has_leaf_right,
                          h_left_child.data(),
                          h_parent.data());
  }

  delete[] h_has_leaf_left;
  delete[] h_has_leaf_right;
}

BENCHMARK(BM_GPU_RadixTree)
    ->UseManualTime()
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

BENCHMARK(BM_CPU_RadixTree)
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->ArgName("NumThreads");
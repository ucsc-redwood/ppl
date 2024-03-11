#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "../config.h"
#include "host_dispatcher.h"
#include "openmp/kernels/00_init.hpp"
#include "openmp/kernels/01_morton.hpp"
#include "openmp/kernels/02_sort.hpp"
#include "openmp/kernels/03_unique.hpp"
#include "openmp/kernels/04_radix_tree.hpp"
#include "openmp/kernels/05_edge_count.hpp"
#include "openmp/kernels/06_prefix_sum.hpp"
#include "openmp/kernels/07_octree.hpp"

namespace bm = benchmark;

static void BM_CPU_Morton(bm::State& st) {
  const auto [n, min_coord, range, _] = configs[0];
  const auto n_threads = static_cast<int>(st.range(0));

  std::vector<glm::vec4> h_points(n);
  std::vector<unsigned int> h_sort(n);

  for (auto _ : st) {
    cpu::k_ComputeMortonCode(
        n_threads, h_points.data(), h_sort.data(), n, min_coord, range);
  }
}

BENCHMARK(BM_CPU_Morton)
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->ArgName("Threads");

// -----------------------------------------------------

static void BM_CPU_Sort(bm::State& st) {
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

static void BM_CPU_Std_Sort(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];

  std::vector<unsigned int> h_sort(n);

  std::generate(
      h_sort.begin(), h_sort.end(), [n = n]() mutable { return --n; });

  for (auto _ : st) {
    std::sort(h_sort.begin(), h_sort.end());
  }
}

BENCHMARK(BM_CPU_Sort)
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->ArgName("Threads");

BENCHMARK(BM_CPU_Std_Sort)->Unit(bm::kMillisecond);

// -----------------------------------------------------

void BM_CPU_Unique(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];

  std::vector<unsigned int> data(n);
  std::iota(data.begin(), data.end(), 0);

  for (auto _ : st) {
    const auto it = std::unique(data.begin(), data.end());
    const auto n_unique = std::distance(data.begin(), it);
    bm::DoNotOptimize(n_unique);
  }
}

BENCHMARK(BM_CPU_Unique)->Unit(bm::kMillisecond);

// -----------------------------------------------------

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

BENCHMARK(BM_CPU_RadixTree)
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->ArgName("NumThreads");

// -----------------------------------------------------

void BM_CPU_EdgeCount(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto n_threads = st.range(0);

  //   std::vector<unsigned int> u_morton(n);
  //   std::iota(u_morton.begin(), u_morton.end(), 0);

  //   const RadixTree brt(n);
  //   std::vector<int> u_edges(n);

  //   cpu::k_BuildRadixTree(n_threads,
  //                         n,
  //                         u_morton.data(),
  //                         brt.u_prefix_n,
  //                         brt.u_has_leaf_left,
  //                         brt.u_has_leaf_right,
  //                         brt.u_left_child,
  //                         brt.u_parent);

  Pipe p(n, min_coord, range, init_seed);

  cpu::k_InitRandomVec4(p.u_points, n, min_coord, range, init_seed);
  cpu::v2::dispatch_ComputeMorton(n_threads, p);
  cpu::v2::dispatch_RadixSort(n_threads, p);
  cpu::v2::dispatch_RemoveDuplicates(n_threads, p);
  cpu::v2::dispatch_BuildRadixTree(n_threads, p);
  cpu::v2::dispatch_EdgeCount(n_threads, p);

  for (auto _ : st) {
    cpu::v2::dispatch_EdgeCount(n_threads, p);
    // cpu::k_EdgeCount(
    //     n_threads, brt.u_prefix_n, brt.u_parent, u_edges.data(), n);
  }
}

BENCHMARK(BM_CPU_EdgeCount)
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->Iterations(300)  // takes too long
    ->ArgName("NumThreads");

// -----------------------------------------------------

void BM_CPU_PrefixSum(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];

  std::vector u_data(n, 1);
  std::vector<int> u_data_out(n);

  for (auto _ : st) {
    std::partial_sum(u_data.begin(), u_data.end(), u_data_out.begin());
  }
}

BENCHMARK(BM_CPU_PrefixSum)->Unit(bm::kMillisecond);

// -----------------------------------------------------

void BM_CPU_Octree(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto n_threads = st.range(0);

  Pipe p(n, min_coord, range, init_seed);

  cpu::k_InitRandomVec4(p.u_points, n, min_coord, range, init_seed);
  cpu::v2::dispatch_ComputeMorton(n_threads, p);
  cpu::v2::dispatch_RadixSort(n_threads, p);
  cpu::v2::dispatch_RemoveDuplicates(n_threads, p);
  cpu::v2::dispatch_BuildRadixTree(n_threads, p);
  cpu::v2::dispatch_EdgeCount(n_threads, p);
  cpu::v2::dispatch_EdgeOffset(n_threads, p);

  for (auto _ : st) {
    cpu::v2::dispatch_BuildOctree(n_threads, p);
  }
}

BENCHMARK(BM_CPU_Octree)
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->Iterations(1)
    ->ArgName("Threads");

BENCHMARK_MAIN();

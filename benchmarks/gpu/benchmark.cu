#include <benchmark/benchmark.h>

#include <numeric>

#include "../config.h"
#include "cu_bench_helper.cuh"
#include "cuda/agents/prefix_sum_agent.cuh"
#include "cuda/agents/unique_agent.cuh"
#include "cuda/kernels/00_init.cuh"
#include "cuda/kernels/01_morton.cuh"
#include "cuda/kernels/02_sort.cuh"
#include "cuda/kernels/03_unique.cuh"
#include "cuda/kernels/04_radix_tree.cuh"
#include "cuda/kernels/05_edge_count.cuh"
#include "cuda/kernels/06_prefix_sum.cuh"
#include "cuda/kernels/07_octree.cuh"
#include "device_dispatcher.h"
#include "handlers/one_sweep.h"
// #include "openmp/kernels/00_init.hpp"

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

BENCHMARK(BM_GPU_Morton)
    ->UseManualTime()
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

// ----------------------------------------------------------------------------

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

BENCHMARK(BM_GPU_Sort)
    ->UseManualTime()
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

// ----------------------------------------------------------------------------

void BM_GPU_Unique(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto grid_size = st.range(0);

  constexpr auto unique_block_size = 256;  // in 'agent.cu'
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
        u_flag_heads, u_flag_heads, n);
    gpu::k_MoveDups<<<grid_size, unique_block_size>>>(
        u_sort, u_flag_heads, n, u_keys_out, nullptr);
  }

  CUDA_FREE(u_sort);
  CUDA_FREE(u_flag_heads);
  CUDA_FREE(u_keys_out);
}

BENCHMARK(BM_GPU_Unique)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

// ----------------------------------------------------------------------------

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

BENCHMARK(BM_GPU_RadixTree)
    ->UseManualTime()
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

// ----------------------------------------------------------------------------

void BM_GPU_EdgeCount(bm::State& st) {
  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto grid_size = st.range(0);

  const auto block_size = determineBlockSizeAndDisplay(gpu::k_EdgeCount, st);

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
        brt.u_prefix_n, brt.u_parent, u_edges, n);
  }

  CUDA_FREE(u_morton);
  CUDA_FREE(u_edges);
}

BENCHMARK(BM_GPU_EdgeCount)
    ->UseManualTime()
    ->Unit(bm::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->Iterations(300)  // takes too long
    ->ArgName("GridSize");

// ----------------------------------------------------------------------------

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

BENCHMARK(BM_GPU_PrefixSum)->UseManualTime()->Unit(bm::kMillisecond);

// ----------------------------------------------------------------------------

void BM_GPU_Octree(bm::State& st) {
  CREATE_STREAM

  const auto [n, min_coord, range, init_seed] = configs[0];
  const auto grid_size = st.range(0);

  Pipe p(n, min_coord, range, init_seed);

  gpu::k_InitRandomVec4<<<64, 256, 0, stream>>>(
      p.u_points, n, min_coord, range, init_seed);
  gpu::v2::dispatch_ComputeMorton(64, stream, p);
  gpu::v2::dispatch_RadixSort(64, stream, p);
  gpu::v2::dispatch_RemoveDuplicates(64, stream, p);
  gpu::v2::dispatch_BuildRadixTree(64, stream, p);
  gpu::v2::dispatch_EdgeCount(64, stream, p);

  SYNC_STREAM(stream);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);

    gpu::v2::dispatch_BuildOctree(grid_size, stream, p);
  }

  DESCROY_STREAM
}

BENCHMARK(BM_GPU_Octree)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->RangeMultiplier(2)
    ->Range(1, 128)
    ->ArgName("GridSize");

// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  int device_count;
  cudaGetDeviceCount(&device_count);

  cudaSetDevice(0);

  for (auto device_id = 0; device_id < device_count; ++device_id) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    std::cout << "Device ID: " << device_id << '\n';
    std::cout << "Device name: " << device_prop.name << '\n';
    std::cout << "Compute capability: " << device_prop.major << "."
              << device_prop.minor << '\n';
    std::cout << "Total global memory: "
              << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << '\n';
    std::cout << "Number of multiprocessors: "
              << device_prop.multiProcessorCount << '\n';
    std::cout << "Max threads per block: " << device_prop.maxThreadsPerBlock
              << '\n';
    std::cout << "Max threads per multiprocessor: "
              << device_prop.maxThreadsPerMultiProcessor << '\n';
    std::cout << "Warp size: " << device_prop.warpSize << '\n';
    std::cout << '\n';
  }

  bm::Initialize(&argc, argv);
  bm::RunSpecifiedBenchmarks();
}
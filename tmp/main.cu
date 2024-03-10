#include <cuda_runtime.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <chrono>

#include "dispatcher.h"

int main(const int argc, const char* argv[]) {
  int n_threads = 1;

  if (argc > 1) {
    n_threads = std::stoi(argv[1]);
  }

  std::cout << "Threads: " << n_threads << std::endl;

  spdlog::set_level(spdlog::level::trace);

  const auto n = 1 << 20;  // 1M
  const auto min_coord = 0.0f;
  const auto range = 1.0f;
  const auto init_seed = 114514;

  const auto pipe = std::make_unique<Pipe>(n, min_coord, range, init_seed);

  cudaStream_t stream;
  CHECK_CUDA_CALL(cudaStreamCreate(&stream));

  omp_set_num_threads(n_threads);
#pragma omp parallel
  { spdlog::debug("Hello from thread {}", omp_get_thread_num()); }

  cpu::k_InitRandomVec4(
      pipe->u_points, pipe->n, pipe->min_coord, pipe->range, pipe->seed);

  cpu::v2::dispatch_ComputeMorton(n_threads, *pipe);
  cpu::v2::dispatch_RadixSort(n_threads, *pipe);
  cpu::v2::dispatch_RemoveDuplicates(n_threads, *pipe);
  cpu::v2::dispatch_BuildRadixTree(n_threads, *pipe);
  cpu::v2::dispatch_EdgeCount(n_threads, *pipe);
  cpu::v2::dispatch_EdgeOffset(n_threads, *pipe);

  // peek 32 prefix n of brt nodes
  for (auto i = 0; i < 32; ++i) {
    spdlog::trace("brt prefix n[{}]: {}", i, pipe->brt.u_prefix_n[i]);
  }

  // peek 32 edge offset
  for (auto i = 0; i < 32; ++i) {
    spdlog::trace("edge offset[{}]: {}", i, pipe->u_edge_offset[i]);
  }

  auto start = std::chrono::high_resolution_clock::now();

  // cpu::v2::dispatch_BuildOctree(n_threads, *pipe);
  gpu::v2::dispatch_BuildOctree(n_threads, stream, *pipe);
  SYNC_STREAM(stream);

  // cpu::k_MakeOctNodes(n_threads,
  //                     pipe->oct.u_children,
  //                     pipe->oct.u_corner,
  //                     pipe->oct.u_cell_size,
  //                     pipe->oct.u_child_node_mask,
  //                     pipe->getEdgeOffset(),
  //                     pipe->getEdgeCount(),
  //                     pipe->getUniqueKeys(),
  //                     pipe->brt.u_prefix_n,
  //                     pipe->brt.u_parent,
  //                     min_coord,
  //                     range,
  //                     pipe->getBrtSize());

  auto end = std::chrono::high_resolution_clock::now();

  // peek 32 octree node u_children

  for (auto i = 0; i < 32; ++i) {
    std::cout << "oct node[" << i << "]: ";
    for (auto j = 0; j < 8; ++j) {
      std::cout << pipe->oct.u_children[i][j] << " ";
    }
    std::cout << '\n';
  }

  spdlog::info(
      "Elapsed time: {} ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count());

  CHECK_CUDA_CALL(cudaStreamDestroy(stream));

  return 0;
}
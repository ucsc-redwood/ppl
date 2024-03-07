#pragma once

#include <glm/glm.hpp>

#include "kernels_fwd.h"
#include "octree.cuh"
#include "one_sweep.cuh"
#include "radix_tree.cuh"
#include "unique.cuh"

struct Pipe {
  // allocate 60% of input size for Octree nodes.
  // From my experiments, the number of octree nodes 45-49% of the number of the
  // input size. Thus, 60% is a safe bet.
  static constexpr auto EDUCATED_GUESS = 0.6;

  const size_t n;
  size_t n_unique_keys;
  size_t n_brt_nodes;
  size_t n_oct_nodes;

  const float min_coord;
  const float range;
  float seed;

  // ------------------------
  // Essential Data
  // ------------------------

  glm::vec4* u_points;
  OneSweepHandler sort;  // u_sort, and temp...
  UniqueHandler unique;  // u_unique_keys, and temp...
  RadixTree brt;         // tree Nodes (SoA) ...
  int* u_edge_count;
  int* u_edge_offset;
  OctreeHandler oct;

  // ------------------------
  Pipe() = delete;

  explicit Pipe(const size_t n,
                const float min_coord,
                const float range,
                const float seed)
      : n(n),
        min_coord(min_coord),
        range(range),
        seed(seed),
        sort(n),
        unique(n),
        brt(n),
        oct(static_cast<size_t>(static_cast<double>(n) * EDUCATED_GUESS)) {
    MALLOC_MANAGED(&u_points, n);
    MALLOC_MANAGED(&u_edge_count, n);
    MALLOC_MANAGED(&u_edge_offset, n);

    SYNC_DEVICE();

    spdlog::trace("On constructor: Pipe, n: {}", n);
  }

  Pipe(const Pipe&) = delete;
  Pipe& operator=(const Pipe&) = delete;
  Pipe(Pipe&&) = delete;
  Pipe& operator=(Pipe&&) = delete;

  ~Pipe() {
    CUDA_FREE(u_points);
    CUDA_FREE(u_edge_count);
    CUDA_FREE(u_edge_offset);

    spdlog::trace("On destructor: Pipe");
  }

  void attachStreamSingle(const cudaStream_t stream) const {
    ATTACH_STREAM_SINGLE(u_points);
    ATTACH_STREAM_SINGLE(u_edge_count);
    ATTACH_STREAM_SINGLE(u_edge_offset);
  }

  void attachStreamGlobal(const cudaStream_t stream) const {
    ATTACH_STREAM_GLOBAL(u_points);
    ATTACH_STREAM_GLOBAL(u_edge_count);
    ATTACH_STREAM_GLOBAL(u_edge_offset);
  }

  void attachStreamHost(const cudaStream_t stream) const {
    ATTACH_STREAM_HOST(u_points);
    ATTACH_STREAM_HOST(u_edge_count);
    ATTACH_STREAM_HOST(u_edge_offset);
  }
};

namespace gpu {
namespace v2 {

[[deprecated]] static void dispatch_Init(const int grid_size,
                                         const cudaStream_t stream,
                                         const Pipe& pipe) {
  constexpr auto block_size = 512;

  spdlog::debug(
      "Dispatching k_InitRandomVec4 with ({} blocks, {} threads) "
      "on {} items",
      grid_size,
      block_size,
      pipe.n);

  k_InitRandomVec4<<<grid_size, block_size, 0, stream>>>(
      pipe.u_points, pipe.n, pipe.min_coord, pipe.range, pipe.seed);
}

static void dispatch_ComputeMorton(const int grid_size,
                                   const cudaStream_t stream,
                                   Pipe& pipe) {
  constexpr auto block_size = 768;

  spdlog::debug(
      "Dispatching k_ComputeMortonCode with ({} blocks, {} "
      "threads) "
      "on {} items",
      grid_size,
      block_size,
      pipe.n);

  k_ComputeMortonCode<<<grid_size, block_size, 0, stream>>>(
      pipe.u_points, pipe.sort.data(), pipe.n, pipe.min_coord, pipe.range);
}

static void dispatch_EdgeCount(const int grid_size,
                               const cudaStream_t stream,
                               const RadixTree& brt,
                               int* edge_count) {
  constexpr auto block_size = 512;

  spdlog::debug(
      "Dispatching k_EdgeCount with ({} blocks, {} threads) "
      "on {} items",
      grid_size,
      block_size,
      brt.getNumBrtNodes());

  k_EdgeCount<<<grid_size, block_size, 0, stream>>>(
      brt.u_prefix_n, brt.u_parent, edge_count, brt.getNumBrtNodes());
}

static void dispatch_EdgeOffset_safe([[maybe_unused]] const int grid_size,
                                     const cudaStream_t stream,
                                     const int* edge_count,
                                     int* edge_offset,
                                     const size_t n_brt_nodes) {
  constexpr auto n_threads = PrefixSumAgent<int>::n_threads;

  spdlog::debug(
      "Dispatching k_SingleBlockExclusiveScan with (1 blocks, {} "
      "threads)",
      n_threads);

  k_SingleBlockExclusiveScan<<<1, n_threads, 0, stream>>>(
      edge_count, edge_offset, n_brt_nodes);
}

}  // namespace v2
}  // namespace gpu

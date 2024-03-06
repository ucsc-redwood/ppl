#pragma once

#include <glm/glm.hpp>

#include "kernels_fwd.h"
#include "one_sweep.cuh"
#include "radix_tree.cuh"

struct Pipe {
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
  OneSweepHandler sort;
  unsigned int* u_unique_keys;
  RadixTree brt;
  int* u_edge_count;
  int* u_edge_offset;

  // ------------------------
  Pipe() = delete;

  explicit Pipe(const size_t n, float min_coord, float range, float seed)
      : n(n), sort(n), brt(n), min_coord(min_coord), range(range), seed(seed) {
    MALLOC_MANAGED(&u_points, n);
    MALLOC_MANAGED(&u_unique_keys, n);
    MALLOC_MANAGED(&u_edge_count, n);
    MALLOC_MANAGED(&u_edge_offset, n);
  }

  Pipe(const Pipe&) = delete;
  Pipe& operator=(const Pipe&) = delete;
  Pipe(Pipe&&) = delete;
  Pipe& operator=(Pipe&&) = delete;

  ~Pipe() {
    CUDA_FREE(u_points);
    CUDA_FREE(u_unique_keys);
    CUDA_FREE(u_edge_count);
    CUDA_FREE(u_edge_offset);
  }

  void attachStreamSingle(const cudaStream_t stream) const {
    ATTACH_STREAM_SINGLE(u_points);
    ATTACH_STREAM_SINGLE(u_unique_keys);
    ATTACH_STREAM_SINGLE(u_edge_count);
    ATTACH_STREAM_SINGLE(u_edge_offset);
  }

  void attachStreamGlobal(const cudaStream_t stream) const {
    ATTACH_STREAM_GLOBAL(u_points);
    ATTACH_STREAM_GLOBAL(u_unique_keys);
    ATTACH_STREAM_GLOBAL(u_edge_count);
    ATTACH_STREAM_GLOBAL(u_edge_offset);
  }

  void attachStreamHost(const cudaStream_t stream) const {
    ATTACH_STREAM_HOST(u_points);
    ATTACH_STREAM_HOST(u_unique_keys);
    ATTACH_STREAM_HOST(u_edge_count);
    ATTACH_STREAM_HOST(u_edge_offset);
  }
};

namespace gpu {
namespace v2 {

static void dispatch_Init(const int grid_size,
                          const cudaStream_t stream,
                          Pipe& pipe) {
  constexpr auto block_size = 512;

  spdlog::debug(
      "Dispatching k_InitRandomVec4 with ({} blocks, {} threads) "
      "on {} items",
      grid_size,
      block_size,
      pipe.n);

  gpu::k_InitRandomVec4<<<grid_size, block_size, 0, stream>>>(
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

  gpu::k_ComputeMortonCode<<<grid_size, block_size, 0, stream>>>(
      pipe.u_points, pipe.sort.data(), pipe.n, pipe.min_coord, pipe.range);
}

}  // namespace v2
}  // namespace gpu

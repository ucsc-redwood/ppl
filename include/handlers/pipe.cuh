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
        n_unique_keys(),
        n_brt_nodes(),
        n_oct_nodes(),
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

  // ------------------------
  // Preferred Getter
  // ------------------------

  [[nodiscard]] size_t getInputSize() const { return n; }
  [[nodiscard]] size_t getUniqueSize() const { return n_unique_keys; }
  [[nodiscard]] size_t getBrtSize() const { return n_brt_nodes; }
  [[nodiscard]] size_t getOctSize() const { return n_oct_nodes; }

  [[nodiscard]] const unsigned int* getSortedKeys() const {
    return sort.u_sort;
  }
  [[nodiscard]] const unsigned int* getUniqueKeys() const {
    return unique.u_keys_out;
  }
  [[nodiscard]] const int* getEdgeCount() const { return u_edge_count; }
  [[nodiscard]] const int* getEdgeOffset() const { return u_edge_offset; }

  // ------------------------

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

#pragma once

#include <algorithm>
#include <glm/common.hpp>

#include "octree.h"
#include "one_sweep.h"
#include "radix_tree.h"
#include "unique.h"

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
  int seed;

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
                const int seed);

  Pipe(const Pipe&) = delete;
  Pipe& operator=(const Pipe&) = delete;
  Pipe(Pipe&&) = delete;
  Pipe& operator=(Pipe&&) = delete;

  ~Pipe();

  // will modify u_points
  void acquireNextFrameData() {
    // rotating the points by 90 degrees.
    // this will modify the position of the points. but making sure the points
    // still in range.
    for (auto i = 0u; i < n; i++) {
      const auto x_tmp = u_points[i].x;
      u_points[i].x = u_points[i].y;
      u_points[i].y = u_points[i].z;
      u_points[i].z = x_tmp;
    }
  };

  // ------------------------
  // Preferred Getter
  // ------------------------

  [[nodiscard]] size_t getInputSize() const { return n; }
  [[nodiscard]] size_t getUniqueSize() const { return n_unique_keys; }
  [[nodiscard]] size_t getBrtSize() const { return n_brt_nodes; }

  [[nodiscard]] size_t getOctSize() const {
    return u_edge_offset[n_brt_nodes - 1];
  }

  [[nodiscard]] const unsigned int* getSortedKeys() const {
    return sort.u_sort;
  }

  [[nodiscard]] const unsigned int* getUniqueKeys() const {
    return unique.u_keys_out;
  }

  [[nodiscard]] const int* getEdgeCount() const { return u_edge_count; }
  [[nodiscard]] const int* getEdgeOffset() const { return u_edge_offset; }
};

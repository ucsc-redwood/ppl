#pragma once

#include <glm/glm.hpp>

#include "brt.cuh"
#include "one_sweep.cuh"
#include "unique.cuh"

struct Variables {
  int n_pts;
  int n_unique_pts;
  int n_brt_nodes;
  int n_oct_nodes;
};

struct Pipe {
  explicit Pipe(const size_t n) : u_points(n), onesweep(n), unique(n), brt(n) {
    MallocManaged(&u_vars, 1);
  }

  ~Pipe() { CUDA_FREE(u_vars); }

  Pipe(const Pipe&) = delete;
  Pipe& operator=(const Pipe&) = delete;
  Pipe(Pipe&&) = delete;
  Pipe& operator=(Pipe&&) = delete;

  void attachStream(const cudaStream_t stream) {
    ATTACH_STREAM_SINGLE(u_vars);
    ATTACH_STREAM_SINGLE(u_points.data());
    onesweep.attachStream(stream);
    unique.attachStream(stream);
    brt.attachStream(stream);
  }

  // clang-format off
  [[nodiscard]] glm::vec4* getPoints() { return u_points.data(); }
  [[nodiscard]] const glm::vec4* getPoints() const { return u_points.data(); }

  [[nodiscard]] unsigned int* getMortonKeys() { return onesweep.getSort(); }
  [[nodiscard]] const unsigned int* getMortonKeys() const { return onesweep.getSort(); }

  // clang-format on

  [[nodiscard]] size_t getMemorySize() const {
    size_t total = 0;
    total += calculateMemorySize(u_points);  // about 32MB
    total += onesweep.getMemorySize();       // about 15MB
    total += unique.getMemorySize();
    total += brt.getMemorySize();  // ???
    return total;
  }

  Variables* u_vars;
  cu::unified_vector<glm::vec4> u_points;
  OneSweep onesweep;
  UniqueKeys unique;
  RadixTree brt;
};

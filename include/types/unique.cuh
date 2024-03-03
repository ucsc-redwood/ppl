#pragma once

#include "cuda/unified_vector.cuh"

struct UniqueKeys {
  explicit UniqueKeys(const size_t n)
      : u_unique_keys(n), u_contributes(n), u_final_pt_idx(n) {}

  ~UniqueKeys() = default;

  UniqueKeys(const UniqueKeys&) = delete;
  UniqueKeys& operator=(const UniqueKeys&) = delete;
  UniqueKeys(UniqueKeys&&) = delete;
  UniqueKeys& operator=(UniqueKeys&&) = delete;

  void attachStream(const cudaStream_t stream) {
    ATTACH_STREAM_SINGLE(u_contributes.data());
    ATTACH_STREAM_SINGLE(u_unique_keys.data());
    ATTACH_STREAM_SINGLE(u_final_pt_idx.data());
  }

  [[nodiscard]] size_t getMemorySize() const {
    size_t total = 0;
    total += calculateMemorySize(u_contributes);
    total += calculateMemorySize(u_unique_keys);
    total += calculateMemorySize(u_final_pt_idx);
    return total;
  }

  cu::unified_vector<unsigned int> u_unique_keys;
  cu::unified_vector<int> u_contributes;
  cu::unified_vector<int> u_final_pt_idx;
};

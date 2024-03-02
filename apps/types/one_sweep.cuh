#pragma once

#include <array>
#include <memory>

#include "cuda/unified_vector.cuh"

// Fixed for 4 passes, 256 radix
// This is what we care, no need to generalize it
//
struct OneSweep {
  static constexpr auto kRadix = 256;
  static constexpr auto kPasses = 4;

  explicit OneSweep(const size_t n)
      : u_sort(n),
        u_sort_alt(n),
        u_global_histogram(kRadix * kPasses),
        u_index(kPasses) {
    for (auto& index : u_pass_histogram) {
      index.resize(kRadix * binningThreadblocks(n));
    }
  }

  OneSweep(const OneSweep&) = delete;
  OneSweep& operator=(const OneSweep&) = delete;
  OneSweep(OneSweep&&) = delete;
  OneSweep& operator=(OneSweep&&) = delete;

  [[nodiscard]] unsigned int* getSort() { return u_sort.data(); }
  [[nodiscard]] const unsigned int* getSort() const { return u_sort.data(); }

  void attachStream(const cudaStream_t stream) {
    ATTACH_STREAM_SINGLE(u_sort.data());
    ATTACH_STREAM_SINGLE(u_sort_alt.data());
    ATTACH_STREAM_SINGLE(u_global_histogram.data());
    ATTACH_STREAM_SINGLE(u_index.data());
    for (auto& pass : u_pass_histogram) {
      ATTACH_STREAM_SINGLE(pass.data());
    }
  }

  [[nodiscard]] size_t getMemorySize() const {
    size_t total = 0;
    total += calculateMemorySize(u_sort);
    total += calculateMemorySize(u_sort_alt);
    total += calculateMemorySize(u_global_histogram);
    total += calculateMemorySize(u_index);
    for (const auto& index : u_pass_histogram) {
      total += calculateMemorySize(index);
    }
    return total;
  }

  static constexpr int binningThreadblocks(const int size) {
    // Need to match to the numbers in the '.cu' file in gpu source code
    constexpr int partitionSize = 7680;
    return (size + partitionSize - 1) / partitionSize;
  }

  static constexpr int globalHistThreadblocks(const int size) {
    // Need to match to the numbers in the '.cu' file in gpu source code
    constexpr int globalHistPartitionSize = 65536;
    return (size + globalHistPartitionSize - 1) / globalHistPartitionSize;
  }

  cu::unified_vector<unsigned int> u_sort;
  cu::unified_vector<unsigned int> u_sort_alt;
  cu::unified_vector<unsigned int> u_global_histogram;
  cu::unified_vector<unsigned int> u_index;
  std::array<cu::unified_vector<unsigned int>, kPasses> u_pass_histogram;
};

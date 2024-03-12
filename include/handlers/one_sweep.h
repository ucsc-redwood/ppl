#pragma once

#include <cstddef>

struct OneSweepHandler {
  // need to match the ".cu" file
  static constexpr auto RADIX = 256;
  static constexpr auto RADIX_PASSES = 4;
  static constexpr auto BIN_PART_SIZE = 7680;
  static constexpr auto BIN_PARTS = 2;
  static constexpr auto GLOBAL_HIST_THREADS = 128;
  static constexpr auto BINNING_THREADS = 512;

  // ------------------------
  // Essential Data
  // ------------------------
  const size_t n;
  const size_t binning_blocks;

  unsigned int* u_sort;
  unsigned int* u_sort_alt;  // cpu might use these

  // ------------------------

  // GPU only
  struct _IntermediateStorage {
    unsigned int* d_global_histogram;
    unsigned int* d_index;
    unsigned int* d_first_pass_histogram;
    unsigned int* d_second_pass_histogram;
    unsigned int* d_third_pass_histogram;
    unsigned int* d_fourth_pass_histogram;
  } im_storage;

  // on purpose, for future work (union the memories)
  using IntermediateStorage = _IntermediateStorage;

  OneSweepHandler() = delete;

  explicit OneSweepHandler(const size_t n);

  OneSweepHandler(const OneSweepHandler&) = delete;
  OneSweepHandler& operator=(const OneSweepHandler&) = delete;
  OneSweepHandler(OneSweepHandler&&) = delete;
  OneSweepHandler& operator=(OneSweepHandler&&) = delete;

  ~OneSweepHandler();

  [[nodiscard]] size_t size() const { return n; }
  [[nodiscard]] const unsigned int* begin() const { return u_sort; }
  [[nodiscard]] unsigned int* begin() { return u_sort; }
  [[nodiscard]] const unsigned int* end() const { return u_sort + n; }
  [[nodiscard]] unsigned int* end() { return u_sort + n; }

  [[nodiscard]] const unsigned int* data() const { return u_sort; }
  [[nodiscard]] unsigned int* data() { return u_sort; }

  void clearMem() const;
};

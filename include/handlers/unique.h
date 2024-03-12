#pragma once

#include <cstddef>

struct UniqueHandler {
  // ------------------------
  // Essential Data
  // ------------------------

  // Inputs:
  // I Should use onesweep's 'u_sort' for input
  const size_t n_input;

  // Output:
  size_t n_unique_keys;
  unsigned int* u_keys_out;

  // Temporary data on device that CPU doesn't need to access
  struct _IntermediateStorage {
    int* u_flag_heads;
  } im_storage;

  using IntermediateStorage = _IntermediateStorage;

  // ------------------------

  UniqueHandler() = delete;

  explicit UniqueHandler(const size_t n_input);

  UniqueHandler(const UniqueHandler&) = delete;
  UniqueHandler& operator=(const UniqueHandler&) = delete;
  UniqueHandler(UniqueHandler&&) = delete;
  UniqueHandler& operator=(UniqueHandler&&) = delete;

  ~UniqueHandler();

  [[nodiscard]] const unsigned int* begin() const { return u_keys_out; }
  [[nodiscard]] const unsigned int* end() const {
    return u_keys_out + n_unique_keys;
  }
};

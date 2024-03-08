#pragma once

#include "cuda/helper.cuh"

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

  explicit UniqueHandler(const size_t n_input)
      : n_input(n_input), n_unique_keys() {
    MALLOC_MANAGED(&u_keys_out, n_input);
    MALLOC_MANAGED(&im_storage.u_flag_heads, n_input);
    // MALLOC_DEVICE(&im_storage.u_flag_heads, n_input);
    SYNC_DEVICE();

    spdlog::trace("On constructor: UniqueHandler, n: {}", n_input);
  }

  UniqueHandler(const UniqueHandler&) = delete;
  UniqueHandler& operator=(const UniqueHandler&) = delete;
  UniqueHandler(UniqueHandler&&) = delete;
  UniqueHandler& operator=(UniqueHandler&&) = delete;

  ~UniqueHandler() {
    CUDA_FREE(u_keys_out);
    CUDA_FREE(im_storage.u_flag_heads);

    spdlog::trace("On destructor: UniqueHandler");
  }

  //   void setNumUnique(const size_t n) { n_unique_keys = n; }
  //   [[nodiscard]] size_t getNumUnique() const { return n_unique_keys; }

  // need to call dispatch_RemoveDuplicates before calling this function
  [[nodiscard]] size_t attemptGetNumUnique() const {
    return im_storage.u_flag_heads[n_input - 1] + 1;
  }

  [[nodiscard]] const unsigned int* begin() const { return u_keys_out; }
  [[nodiscard]] const unsigned int* end() const {
    return u_keys_out + n_unique_keys;
  }

  void attachStreamSingle(const cudaStream_t stream) const {
    ATTACH_STREAM_SINGLE(u_keys_out);
    ATTACH_STREAM_SINGLE(im_storage.u_flag_heads);
  }

  void attachStreamGlobal(const cudaStream_t stream) const {
    ATTACH_STREAM_GLOBAL(u_keys_out);
    ATTACH_STREAM_GLOBAL(im_storage.u_flag_heads);
  }

  void attachStreamHost(const cudaStream_t stream) const {
    ATTACH_STREAM_HOST(u_keys_out);
    ATTACH_STREAM_HOST(im_storage.u_flag_heads);
  }
};

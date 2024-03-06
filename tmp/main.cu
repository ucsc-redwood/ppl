#include <cuda_runtime.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <memory>
#include <type_traits>

#include "cuda/kernels/02_sort.cuh"
#include "cuda/unified_vector.cuh"

struct OneSweepHandler {
  // need to match the ".cu" file
  static constexpr auto RADIX = 256;
  static constexpr auto RADIX_PASSES = 4;
  static constexpr auto BIN_PART_SIZE = 7680;
  static constexpr auto BIN_PARTS = 2;

  static constexpr auto GLOBAL_HIST_THREADS = 128;
  static constexpr auto BINNING_THREADS = 512;

  // Essential Data
  const size_t n;
  const size_t binning_blocks;  // abpit n / 7680

  unsigned int* u_sort;

  struct _IntermediateStorage {
    unsigned int* d_sort_alt;
    unsigned int* d_global_histogram;
    unsigned int* d_index;
    unsigned int* d_first_pass_histogram;
    unsigned int* d_second_pass_histogram;
    unsigned int* d_third_pass_histogram;
    unsigned int* d_fourth_pass_histogram;
  } im_storage;

  using IntermediateStorage = _IntermediateStorage;

  OneSweepHandler() = delete;

  explicit OneSweepHandler(const size_t n)
      : n(n), binning_blocks(cub::DivideAndRoundUp(n, BIN_PART_SIZE)) {
    // Essential buffer that CPU/GPU both can access
    MALLOC_MANAGED(&u_sort, n);

    // Temporary data on device that CPU doesn't need to access
    MALLOC_DEVICE(&im_storage.d_sort_alt, n);
    MALLOC_DEVICE(&im_storage.d_global_histogram, RADIX * RADIX_PASSES);
    MALLOC_DEVICE(&im_storage.d_index, RADIX_PASSES);
    MALLOC_DEVICE(&im_storage.d_first_pass_histogram, RADIX * binning_blocks);
    MALLOC_DEVICE(&im_storage.d_second_pass_histogram, RADIX * binning_blocks);
    MALLOC_DEVICE(&im_storage.d_third_pass_histogram, RADIX * binning_blocks);
    MALLOC_DEVICE(&im_storage.d_fourth_pass_histogram, RADIX * binning_blocks);
    SYNC_DEVICE();
  }

  OneSweepHandler(const OneSweepHandler&) = delete;
  OneSweepHandler& operator=(const OneSweepHandler&) = delete;
  OneSweepHandler(OneSweepHandler&&) = delete;
  OneSweepHandler& operator=(OneSweepHandler&&) = delete;

  ~OneSweepHandler() {
    CUDA_FREE(u_sort);
    CUDA_FREE(im_storage.d_sort_alt);
    CUDA_FREE(im_storage.d_global_histogram);
    CUDA_FREE(im_storage.d_index);
    CUDA_FREE(im_storage.d_first_pass_histogram);
    CUDA_FREE(im_storage.d_second_pass_histogram);
    CUDA_FREE(im_storage.d_third_pass_histogram);
    CUDA_FREE(im_storage.d_fourth_pass_histogram);
  }

  [[nodiscard]] const unsigned int* begin() const { return u_sort; }
  [[nodiscard]] unsigned int* begin() { return u_sort; }
  [[nodiscard]] const unsigned int* end() const { return u_sort + n; }
  [[nodiscard]] unsigned int* end() { return u_sort + n; }

  [[nodiscard]] const unsigned int* getSort() const { return u_sort; }
  [[nodiscard]] unsigned int* getSort() { return u_sort; }

  void attachStreamSingle(const cudaStream_t stream) const {
    ATTACH_STREAM_SINGLE(u_sort);
  }

  void attachStreamGlobal(const cudaStream_t stream) const {
    ATTACH_STREAM_GLOBAL(u_sort);
  }

  void attachStreamHost(const cudaStream_t stream) const {
    ATTACH_STREAM_HOST(u_sort);
    SYNC_STREAM(stream);
  }

  void memoryUsage() const {
    const auto essential = CALC_MEM(u_sort, n);

    auto temp = 0;
    temp += CALC_MEM(im_storage.d_sort_alt, n);
    temp += CALC_MEM(im_storage.d_global_histogram, RADIX * RADIX_PASSES);
    temp += CALC_MEM(im_storage.d_index, RADIX_PASSES);
    temp += CALC_MEM(im_storage.d_first_pass_histogram, RADIX * binning_blocks);
    temp +=
        CALC_MEM(im_storage.d_second_pass_histogram, RADIX * binning_blocks);
    temp += CALC_MEM(im_storage.d_third_pass_histogram, RADIX * binning_blocks);
    temp +=
        CALC_MEM(im_storage.d_fourth_pass_histogram, RADIX * binning_blocks);

    std::cout << "Total: " << (essential + temp) / 1024.0 / 1024.0
              << " MB (100.000%) Essential: " << essential / 1024.0 / 1024.0
              << " MB (" << (essential / (essential + temp)) * 100.0
              << "%) Temporary: " << temp / 1024.0 / 1024.0 << " MB ("
              << (temp / (essential + temp)) * 100.0 << "%)" << std::endl;
  }

  void clearMem() const {
    SET_MEM_2_ZERO(im_storage.d_global_histogram, RADIX * RADIX_PASSES);
    SET_MEM_2_ZERO(im_storage.d_index, RADIX_PASSES);
    SET_MEM_2_ZERO(im_storage.d_first_pass_histogram, RADIX * binning_blocks);
    SET_MEM_2_ZERO(im_storage.d_second_pass_histogram, RADIX * binning_blocks);
    SET_MEM_2_ZERO(im_storage.d_third_pass_histogram, RADIX * binning_blocks);
    SET_MEM_2_ZERO(im_storage.d_fourth_pass_histogram, RADIX * binning_blocks);
  }
};

int main(const int argc, const char* const argv[]) {
  static_assert(sizeof(unsigned int) == 4, "sizeof(unsigned int) != 4");

  constexpr int n = 1 << 20;  // 1M elements

  auto grid_size = 1;
  auto n_iterations = 2;

  if (argc > 1) {
    grid_size = std::strtol(argv[1], nullptr, 10);
  }

  if (argc > 2) {
    n_iterations = std::strtol(argv[2], nullptr, 10);
  }

  int device = 0;
  cudaDeviceProp prop;
  CHECK_CUDA_CALL(cudaGetDeviceProperties(&prop, device));
  std::cout << "Concurrent Managed Access: " << prop.concurrentManagedAccess
            << '\n';

  std::cout << "Number of elements: " << n << '\n';
  std::cout << "Grid size: " << grid_size << '\n';
  std::cout << "Number of iterations: " << n_iterations << '\n';

  const auto handler = std::make_unique<OneSweepHandler>(n);

  cudaStream_t stream;
  CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  handler->memoryUsage();

  cudaEvent_t start, stop;

  CHECK_CUDA_CALL(cudaEventCreate(&start));
  CHECK_CUDA_CALL(cudaEventCreate(&stop));

  WarmUpGPU();

  CHECK_CUDA_CALL(cudaEventRecord(start, stream));

  for (auto i = 0; i < n_iterations; ++i) {
    // input
    std::generate(
        handler->begin(), handler->end(), [i = n]() mutable { return --i; });

    auto is_sorted = std::is_sorted(handler->begin(), handler->end());
    std::cout << "Is sorted (before): " << std::boolalpha << is_sorted << '\n';

    // ------------------------------
    handler->clearMem();
    handler->attachStreamSingle(stream);

    gpu::k_GlobalHistogram<<<grid_size,
                             OneSweepHandler::GLOBAL_HIST_THREADS,
                             0,
                             stream>>>(
        handler->u_sort, handler->im_storage.d_global_histogram, n);

    gpu::k_Scan<<<OneSweepHandler::RADIX_PASSES,
                  OneSweepHandler::RADIX,
                  0,
                  stream>>>(handler->im_storage.d_global_histogram,
                            handler->im_storage.d_first_pass_histogram,
                            handler->im_storage.d_second_pass_histogram,
                            handler->im_storage.d_third_pass_histogram,
                            handler->im_storage.d_fourth_pass_histogram);

    gpu::k_DigitBinningPass<<<grid_size,
                              OneSweepHandler::BINNING_THREADS,
                              0,
                              stream>>>(
        handler->u_sort,  // <---
        handler->im_storage.d_sort_alt,
        handler->im_storage.d_first_pass_histogram,
        handler->im_storage.d_index,
        n,
        0);
    gpu::k_DigitBinningPass<<<grid_size,
                              OneSweepHandler::BINNING_THREADS,
                              0,
                              stream>>>(
        handler->im_storage.d_sort_alt,
        handler->u_sort,  // <---
        handler->im_storage.d_second_pass_histogram,
        handler->im_storage.d_index,
        n,
        8);
    gpu::k_DigitBinningPass<<<grid_size,
                              OneSweepHandler::BINNING_THREADS,
                              0,
                              stream>>>(
        handler->u_sort,  // <---
        handler->im_storage.d_sort_alt,
        handler->im_storage.d_third_pass_histogram,
        handler->im_storage.d_index,
        n,
        16);
    gpu::k_DigitBinningPass<<<grid_size,
                              OneSweepHandler::BINNING_THREADS,
                              0,
                              stream>>>(
        handler->im_storage.d_sort_alt,
        handler->u_sort,  // <---
        handler->im_storage.d_fourth_pass_histogram,
        handler->im_storage.d_index,
        n,
        24);

    // ------------------------------

    SYNC_STREAM(stream);
    is_sorted = std::is_sorted(handler->begin(), handler->end());
    std::cout << "Is sorted (after): " << std::boolalpha << is_sorted << '\n';
  }

  CHECK_CUDA_CALL(cudaEventRecord(stop, stream));
  CHECK_CUDA_CALL(cudaEventSynchronize(stop));

  float milliseconds = 0;
  // Print total time and average time per iteration
  CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "Total time: " << milliseconds << " ms\n";
  std::cout << "Average time per iteration: " << milliseconds / n_iterations
            << " ms\n";

  CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  return 0;
}
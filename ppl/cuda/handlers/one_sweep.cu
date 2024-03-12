#include <spdlog/spdlog.h>

#include <cub/cub.cuh>

#include "cuda/helper.cuh"
#include "handlers/one_sweep.h"

OneSweepHandler::OneSweepHandler(const size_t n)
    : n(n), binning_blocks(cub::DivideAndRoundUp(n, BIN_PART_SIZE)) {
  // Essential buffer that CPU/GPU both can access
  MALLOC_MANAGED(&u_sort, n);
  MALLOC_MANAGED(&u_sort_alt, n);

  // Temporary data on device that CPU doesn't need to access
  MALLOC_DEVICE(&im_storage.d_global_histogram, RADIX * RADIX_PASSES);
  MALLOC_DEVICE(&im_storage.d_index, RADIX_PASSES);
  MALLOC_DEVICE(&im_storage.d_first_pass_histogram, RADIX * binning_blocks);
  MALLOC_DEVICE(&im_storage.d_second_pass_histogram, RADIX * binning_blocks);
  MALLOC_DEVICE(&im_storage.d_third_pass_histogram, RADIX * binning_blocks);
  MALLOC_DEVICE(&im_storage.d_fourth_pass_histogram, RADIX * binning_blocks);
  SYNC_DEVICE();

  spdlog::trace("On constructor: OneSweepHandler, n: {}, binning_blocks: {}, ",
                n,
                binning_blocks);
}

OneSweepHandler::~OneSweepHandler() {
  CUDA_FREE(u_sort);
  CUDA_FREE(u_sort_alt);
  CUDA_FREE(im_storage.d_global_histogram);
  CUDA_FREE(im_storage.d_index);
  CUDA_FREE(im_storage.d_first_pass_histogram);
  CUDA_FREE(im_storage.d_second_pass_histogram);
  CUDA_FREE(im_storage.d_third_pass_histogram);
  CUDA_FREE(im_storage.d_fourth_pass_histogram);

  spdlog::trace("On destructor: OneSweepHandler");
}

void OneSweepHandler::clearMem() const {
  SET_MEM_2_ZERO(im_storage.d_global_histogram, RADIX * RADIX_PASSES);
  SET_MEM_2_ZERO(im_storage.d_index, RADIX_PASSES);
  SET_MEM_2_ZERO(im_storage.d_first_pass_histogram, RADIX * binning_blocks);
  SET_MEM_2_ZERO(im_storage.d_second_pass_histogram, RADIX * binning_blocks);
  SET_MEM_2_ZERO(im_storage.d_third_pass_histogram, RADIX * binning_blocks);
  SET_MEM_2_ZERO(im_storage.d_fourth_pass_histogram, RADIX * binning_blocks);
}

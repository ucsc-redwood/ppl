#include "handlers/one_sweep.h"

#include <spdlog/spdlog.h>

#include "shared/utils.h"

OneSweepHandler::OneSweepHandler(const size_t n)
    : n(n), binning_blocks(DivideAndRoundUp(n, BIN_PART_SIZE)) {
  u_sort = new unsigned int[n];
  u_sort_alt = new unsigned int[n];

  im_storage.d_global_histogram = new unsigned int[RADIX * RADIX_PASSES];
  im_storage.d_index = new unsigned int[RADIX_PASSES];
  im_storage.d_first_pass_histogram = new unsigned int[RADIX * binning_blocks];
  im_storage.d_second_pass_histogram = new unsigned int[RADIX * binning_blocks];
  im_storage.d_third_pass_histogram = new unsigned int[RADIX * binning_blocks];
  im_storage.d_fourth_pass_histogram = new unsigned int[RADIX * binning_blocks];

  spdlog::trace("On constructor: OneSweepHandler, n: {}, binning_blocks: {}, ",
                n,
                binning_blocks);
}

OneSweepHandler::~OneSweepHandler() {
  delete[] u_sort;
  delete[] u_sort_alt;
  delete[] im_storage.d_global_histogram;
  delete[] im_storage.d_index;
  delete[] im_storage.d_first_pass_histogram;
  delete[] im_storage.d_second_pass_histogram;
  delete[] im_storage.d_third_pass_histogram;
  delete[] im_storage.d_fourth_pass_histogram;

  spdlog::trace("On destructor: OneSweepHandler");
}

#define SET_MEM_2_ZERO(ptr, size) \
  std::memset(ptr, 0, size * sizeof(decltype(*ptr)))

void OneSweepHandler::clearMem() const {
  SET_MEM_2_ZERO(im_storage.d_global_histogram, RADIX * RADIX_PASSES);
  SET_MEM_2_ZERO(im_storage.d_index, RADIX_PASSES);
  SET_MEM_2_ZERO(im_storage.d_first_pass_histogram, RADIX * binning_blocks);
  SET_MEM_2_ZERO(im_storage.d_second_pass_histogram, RADIX * binning_blocks);
  SET_MEM_2_ZERO(im_storage.d_third_pass_histogram, RADIX * binning_blocks);
  SET_MEM_2_ZERO(im_storage.d_fourth_pass_histogram, RADIX * binning_blocks);
}

#include <spdlog/spdlog.h>

#include "cuda/helper.cuh"
#include "handlers/unique.h"

UniqueHandler::UniqueHandler(const size_t n_input)
    : n_input(n_input), n_unique_keys() {
  MALLOC_MANAGED(&u_keys_out, n_input);
  MALLOC_MANAGED(&im_storage.u_flag_heads, n_input);

  SYNC_DEVICE();

  spdlog::trace("On constructor: UniqueHandler, n: {}", n_input);
}

UniqueHandler::~UniqueHandler() {
  CUDA_FREE(u_keys_out);
  CUDA_FREE(im_storage.u_flag_heads);

  spdlog::trace("On destructor: UniqueHandler");
}

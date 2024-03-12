#include "handlers/unique.h"

#include <spdlog/spdlog.h>

UniqueHandler::UniqueHandler(const size_t n_input)
    : n_input(n_input), n_unique_keys() {
  u_keys_out = new unsigned int[n_input];
  im_storage.u_flag_heads = new int[n_input];

  spdlog::trace("On constructor: UniqueHandler, n: {}", n_input);
}

UniqueHandler::~UniqueHandler() {
  delete[] u_keys_out;
  delete[] im_storage.u_flag_heads;

  spdlog::trace("On destructor: UniqueHandler");
}

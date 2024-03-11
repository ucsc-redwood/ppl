#include <omp.h>
#include <spdlog/spdlog.h>

#include "../app_params.hpp"
#include "handlers/pipe.h"
#include "host_dispatcher.h"

int main(const int argc, const char* argv[]) {
  AppParams params(argc, argv);
  params.print_params();

  switch (params.log_level) {
    case 0:
      spdlog::set_level(spdlog::level::off);
      break;
    case 1:
      spdlog::set_level(spdlog::level::info);
      break;
    case 2:
      spdlog::set_level(spdlog::level::debug);
      break;
    case 3:
      spdlog::set_level(spdlog::level::trace);
      break;
    default:
      spdlog::set_level(spdlog::level::info);
      break;
  }

  omp_set_num_threads(params.n_threads);
#pragma omp parallel
  { spdlog::debug("Hello from thread {}", omp_get_thread_num()); }

  return 0;
}
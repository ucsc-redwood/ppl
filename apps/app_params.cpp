#include "app_params.hpp"

#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

AppParams::AppParams(const int argc, const char **argv) {
  CLI::App app{"App description"};

  app.add_option("-n,--n", n, "Number of points")->check(CLI::PositiveNumber);

  app.add_option("-t,--threads", n_threads, "Number of CPU threads")
      ->check(CLI::Range(1, 32));

  app.add_option("-b,--blocks", n_blocks, "Number of GPU blocks")
      ->check(CLI::Range(1, 128));

  app.add_option("-m,--method", method, "Method")->check(CLI::Range(0, 2));

  app.add_option("-l,--log-level", log_level, "Log level")
      ->check(CLI::Range(0, 3));

  app.add_option("-i,--iterations", n_iterations, "Number of iterations")
      ->check(CLI::PositiveNumber);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    (app).exit(e);
    exit(EXIT_FAILURE);
  }
}

constexpr auto method_name = [](int method) {
  switch (method) {
    case 0:
      return "GPU-only";
    case 1:
      return "BestOnEachPU";
    case 2:
      return "TwoPhaseCoarseGrained";
    default:
      return "Unknown";
  }
};

void AppParams::print_params() const {
  spdlog::info("n: {}", n);
  spdlog::info("min_coord: {}", min_coord);
  spdlog::info("max_coord: {}", max_coord);
  spdlog::info("seed: {}", seed);
  spdlog::info("n_threads: {}", n_threads);
  spdlog::info("n_blocks: {}", n_blocks);
  spdlog::info("method: {}", method_name(method));
}
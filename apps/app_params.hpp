#pragma once

struct AppParams {
  // [[nodiscard]] static AppParams from_args(int argc, const char **argv);

  explicit AppParams(int argc, const char** argv);

  [[nodiscard]] float getRange() const { return max_coord - min_coord; }

  void print_params() const;

  // Problem size parameters
  int n = 1920 * 1080;
  float min_coord = 0.0f;
  float max_coord = 1024.0f;
  int seed = 114514;

  // Execution parameters
  int n_threads = 4;
  int n_blocks = 64;

  // Debug parameters
  bool debug_print = false;
  bool use_cpu = false;
};

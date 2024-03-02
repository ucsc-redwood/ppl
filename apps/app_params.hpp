#pragma once

struct AppParams {
  // [[nodiscard]] static AppParams from_args(int argc, char **argv);

  [[nodiscard]] float getRange() { return max_coord - min_coord; }

  int n;
  float min_coord;
  float max_coord;
  int seed;
  int n_threads;
  int n_blocks;
};

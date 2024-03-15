#pragma once

#include <array>
#include <cstddef>

struct Config {
  size_t n;
  float min_coord;
  float range;
  int init_seed;
};

static constexpr std::array<Config, 2> configs{{
    {1 << 20, 0.0f, 1024.0f, 114514},
    {1920 * 1080, 0.0f, 1024.0f, 1919810},
}};

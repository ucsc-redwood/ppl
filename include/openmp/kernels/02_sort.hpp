#pragma once

#include <algorithm>

namespace cpu {

inline void std_sort(unsigned int *sort,
                     [[maybe_unused]] unsigned int *alt,
                     const unsigned int size) {
  std::sort(sort, sort + size);
}

inline void std_sort(unsigned int *sort, const unsigned int size) {
  std_sort(sort, nullptr, size);
}

void k_Sort(int num_threads,
            unsigned int *u_sort,
            unsigned int *u_sort_alt,
            size_t n);

}  // namespace cpu
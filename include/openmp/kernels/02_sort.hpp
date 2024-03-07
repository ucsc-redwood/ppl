#pragma once

namespace cpu {

void k_Sort(int num_threads,
            unsigned int *u_sort,
            unsigned int *u_sort_alt,
            size_t n);

}  // namespace cpu
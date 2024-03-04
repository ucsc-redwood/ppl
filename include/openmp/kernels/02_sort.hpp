#pragma once

#include <algorithm>

namespace cpu {

// ============================================================================
// Kernel entry points
// ============================================================================

inline void std_sort(unsigned int *sort, unsigned int *alt, unsigned int size) {
  std::sort(sort, sort + size);
}

inline void std_sort(unsigned int *sort, unsigned int size) {
  std_sort(sort, nullptr, size);
}

// __global__ void k_GlobalHistogram(unsigned int *sort,
//                                   unsigned int *global_histogram,
//                                   unsigned int size);

// __global__ void k_Scan(unsigned int *globalHistogram,
//                        unsigned int *firstPassHistogram,
//                        unsigned int *secPassHistogram,
//                        unsigned int *thirdPassHistogram,
//                        unsigned int *fourthPassHistogram);

// __global__ void k_DigitBinningPass(unsigned int *sort,
//                                    unsigned int *alt,
//                                    volatile unsigned int *passHistogram,
//                                    volatile unsigned int *index,
//                                    unsigned int size,
//                                    unsigned int radixShift);

}  // namespace cpu
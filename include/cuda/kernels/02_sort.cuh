#pragma once

namespace gpu {

// ============================================================================
// Kernel entry points
// ============================================================================

__global__ void k_GlobalHistogram(const unsigned int *sort,
                                  unsigned int *global_histogram,
                                  unsigned int size);

__global__ void k_Scan(const unsigned int *globalHistogram,
                       unsigned int *firstPassHistogram,
                       unsigned int *secPassHistogram,
                       unsigned int *thirdPassHistogram,
                       unsigned int *fourthPassHistogram);

__global__ void k_DigitBinningPass(unsigned int *sort,
                                   unsigned int *alt,
                                   volatile unsigned int *passHistogram,
                                   volatile unsigned int *index,
                                   unsigned int size,
                                   unsigned int radixShift);

namespace backup {
__global__ void k_DigitBinningPass_Original(
    const unsigned int *sort,
    unsigned int *alt,
    volatile unsigned int *passHistogram,
    volatile unsigned int *index,
    unsigned int size,
    unsigned int radixShift);
}

}  // namespace gpu
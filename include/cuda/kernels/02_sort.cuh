#pragma once

#include <cuda_runtime_api.h>

namespace gpu {

__global__ void k_GlobalHistogram(unsigned int *sort,
                                  unsigned int *globalHistogram,
                                  unsigned int size);

__global__ void k_Scan(unsigned int *globalHistogram,
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

}  // namespace gpu
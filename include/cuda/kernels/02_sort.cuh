#pragma once

#include <cstdint>

namespace gpu {

__global__ void k_GlobalHistogram_WithLogicalBlocks(uint32_t *sort,
                                                    uint32_t *globalHistogram,
                                                    uint32_t size,
                                                    int logicalBlocks);
__global__ void k_DigitBinning_WithLogicalBlocks(
    uint32_t *globalHistogram,
    uint32_t *sort,
    uint32_t *alt,
    volatile uint32_t *passHistogram,
    uint32_t *index,
    uint32_t size,
    uint32_t radixShift,
    int logicalBlocks);

[[deprecated("use the k_GlobalHistogram_WithLogicalBlocks")]] __global__ void
k_GlobalHistogram(uint32_t *sort, uint32_t *globalHistogram, uint32_t size);

[[deprecated("use the k_DigitBinning_WithLogicalBlocks")]] __global__ void
k_DigitBinning(uint32_t *globalHistogram,
               uint32_t *sort,
               uint32_t *alt,
               volatile uint32_t *passHistogram,
               uint32_t *index,
               uint32_t size,
               uint32_t radixShift);

}  // namespace gpu
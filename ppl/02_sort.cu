/******************************************************************************
 * OneSweep
 *
 * Author:  Thomas Smith 9/13/2023
 *
 * Based off of Research by:
 *          Andy Adinets, Nvidia Corporation
 *          Duane Merrill, Nvidia Corporation
 *          https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus
 *
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/util_math.cuh>

#include "cuda/kernels/common.cuh"

namespace gpu {
#define RADIX 256       // Number of digit bins
#define RADIX_MASK 255  // Mask of digit bins, to extract digits
#define RADIX_LOG 8     // log2(RADIX)

#define SEC_RADIX_START 256
#define THIRD_RADIX_START 512
#define FOURTH_RADIX_START 768

// For the upfront global histogram kernel
#define G_HIST_PART_SIZE 65536
#define G_HIST_VEC_SIZE 16384

// For the digit binning
#define BIN_PART_SIZE 7680  // Partition tile size in k_DigitBinning

#define BIN_HISTS_SIZE \
  4096  // Total size of warp histograms in shared memory in k_DigitBinning

#define BIN_SUB_PART_SIZE \
  480  // Subpartition tile size of a single warp in k_DigitBinning

#define BIN_WARPS 16  // Warps per threadblock in k_DigitBinning

#define BIN_KEYS_PER_THREAD 15  // Keys per thread in k_DigitBinning

#define BIN_SUB_PART_START \
  (WARP_INDEX * BIN_SUB_PART_SIZE)  // Starting offset of a subpartition tile

#define BIN_PART_START \
  (partitionIndex * BIN_PART_SIZE)  // Starting offset of a partition tile

// for the chained scan with decoupled lookback
#define FLAG_NOT_READY \
  0  // Flag value inidicating neither inclusive sum, nor reduction of a
     // partition tile is ready

#define FLAG_REDUCTION \
  1  // Flag value indicating reduction of a partition tile is ready

#define FLAG_INCLUSIVE \
  2  // Flag value indicating inclusive sum of a partition tile is ready

#define FLAG_MASK 3

__global__ void k_GlobalHistogram(unsigned int* sort,
                                  unsigned int* global_histogram,
                                  const unsigned int size) {
  __shared__ unsigned int s_globalHistFirst[RADIX * 2];
  __shared__ unsigned int s_globalHistSec[RADIX * 2];
  __shared__ unsigned int s_globalHistThird[RADIX * 2];
  __shared__ unsigned int s_globalHistFourth[RADIX * 2];

  const auto logicalBlocks = cub::DivideAndRoundUp(size, G_HIST_PART_SIZE);

  for (auto yanwen_block_id = blockIdx.x; yanwen_block_id < logicalBlocks;
       yanwen_block_id += gridDim.x) {
    // clear shared memory
    for (unsigned int i = threadIdx.x; i < RADIX * 2; i += blockDim.x) {
      s_globalHistFirst[i] = 0;
      s_globalHistSec[i] = 0;
      s_globalHistThird[i] = 0;
      s_globalHistFourth[i] = 0;
    }
    __syncthreads();

    // histogram
    {
      // 64 threads : 1 histogram in shared memory
      unsigned int* s_wavesHistFirst =
          &s_globalHistFirst[threadIdx.x / 64 * RADIX];
      unsigned int* s_wavesHistSec = &s_globalHistSec[threadIdx.x / 64 * RADIX];
      unsigned int* s_wavesHistThird =
          &s_globalHistThird[threadIdx.x / 64 * RADIX];
      unsigned int* s_wavesHistFourth =
          &s_globalHistFourth[threadIdx.x / 64 * RADIX];

      if (yanwen_block_id < logicalBlocks - 1) {
        const unsigned int partEnd = (yanwen_block_id + 1) * G_HIST_VEC_SIZE;
        for (unsigned int i = threadIdx.x + (yanwen_block_id * G_HIST_VEC_SIZE);
             i < partEnd;
             i += blockDim.x) {
          uint4 t[1] = {reinterpret_cast<uint4*>(sort)[i]};

          atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[0]], 1);
          atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[1]], 1);
          atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t*>(t)[2]], 1);
          atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[3]], 1);

          atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[4]], 1);
          atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[5]], 1);
          atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t*>(t)[6]], 1);
          atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[7]], 1);

          atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[8]], 1);
          atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[9]], 1);
          atomicAdd(&s_globalHistThird[reinterpret_cast<uint8_t*>(t)[10]], 1);
          atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[11]], 1);

          atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[12]], 1);
          atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[13]], 1);
          atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t*>(t)[14]], 1);
          atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[15]], 1);
        }
      }

      if (yanwen_block_id == logicalBlocks - 1) {
        for (unsigned int i =
                 threadIdx.x + (yanwen_block_id * G_HIST_PART_SIZE);
             i < size;
             i += blockDim.x) {
          unsigned int t[1] = {sort[i]};
          atomicAdd(&s_wavesHistFirst[reinterpret_cast<uint8_t*>(t)[0]], 1);
          atomicAdd(&s_wavesHistSec[reinterpret_cast<uint8_t*>(t)[1]], 1);
          atomicAdd(&s_wavesHistThird[reinterpret_cast<uint8_t*>(t)[2]], 1);
          atomicAdd(&s_wavesHistFourth[reinterpret_cast<uint8_t*>(t)[3]], 1);
        }
      }
    }
    __syncthreads();

    // reduce and add to device
    for (unsigned int i = threadIdx.x; i < RADIX; i += blockDim.x) {
      atomicAdd(&global_histogram[i],
                s_globalHistFirst[i] + s_globalHistFirst[i + RADIX]);
      atomicAdd(&global_histogram[i + SEC_RADIX_START],
                s_globalHistSec[i] + s_globalHistSec[i + RADIX]);
      atomicAdd(&global_histogram[i + THIRD_RADIX_START],
                s_globalHistThird[i] + s_globalHistThird[i + RADIX]);
      atomicAdd(&global_histogram[i + FOURTH_RADIX_START],
                s_globalHistFourth[i] + s_globalHistFourth[i + RADIX]);
    }
  }
}

// fixed to use 4 blocks, and 'radix' (256) threads
__global__ void k_Scan(unsigned int* globalHistogram,
                       unsigned int* firstPassHistogram,
                       unsigned int* secPassHistogram,
                       unsigned int* thirdPassHistogram,
                       unsigned int* fourthPassHistogram) {
  __shared__ unsigned int s_scan[RADIX];

  s_scan[threadIdx.x] = InclusiveWarpScanCircularShift(
      globalHistogram[threadIdx.x + blockIdx.x * RADIX]);
  __syncthreads();

  if (threadIdx.x < (RADIX >> LANE_LOG))
    s_scan[threadIdx.x << LANE_LOG] =
        ActiveExclusiveWarpScan(s_scan[threadIdx.x << LANE_LOG]);
  __syncthreads();

  switch (blockIdx.x) {
    case 0:
      firstPassHistogram[threadIdx.x] =
          (s_scan[threadIdx.x] +
           (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1)
                        : 0))
              << 2 |
          FLAG_INCLUSIVE;
      break;
    case 1:
      secPassHistogram[threadIdx.x] =
          (s_scan[threadIdx.x] +
           (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1)
                        : 0))
              << 2 |
          FLAG_INCLUSIVE;
      break;
    case 2:
      thirdPassHistogram[threadIdx.x] =
          (s_scan[threadIdx.x] +
           (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1)
                        : 0))
              << 2 |
          FLAG_INCLUSIVE;
      break;
    case 3:
      fourthPassHistogram[threadIdx.x] =
          (s_scan[threadIdx.x] +
           (getLaneId() ? __shfl_sync(0xfffffffe, s_scan[threadIdx.x - 1], 1)
                        : 0))
              << 2 |
          FLAG_INCLUSIVE;
      break;
    default:
      break;
  }
}

__global__ void k_DigitBinningPass_Original(
    unsigned int* sort,
    unsigned int* alt,
    volatile unsigned int* passHistogram,
    volatile unsigned int* index,
    unsigned int size,
    unsigned int radixShift) {
  __shared__ unsigned int s_warpHistograms[BIN_PART_SIZE];
  __shared__ unsigned int s_localHistogram[RADIX];
  volatile unsigned int* s_warpHist =
      &s_warpHistograms[WARP_INDEX << RADIX_LOG];

  // clear shared memory
  for (unsigned int i = threadIdx.x; i < BIN_HISTS_SIZE;
       i += blockDim.x)  // unnecessary work for last partion but still a win to
                         // avoid another barrier
    s_warpHistograms[i] = 0;

  // atomically assign partition tiles
  if (threadIdx.x == 0)
    s_warpHistograms[BIN_PART_SIZE - 1] =
        atomicAdd((unsigned int*)&index[radixShift >> 3], 1);
  __syncthreads();
  const unsigned int partitionIndex = s_warpHistograms[BIN_PART_SIZE - 1];

  // To handle input sizes not perfect multiples of the partition tile size
  if (partitionIndex < gridDim.x - 1) {
    // load keys
    unsigned int keys[BIN_KEYS_PER_THREAD];
#pragma unroll
    for (unsigned int i = 0,
                      t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;
         i < BIN_KEYS_PER_THREAD;
         ++i, t += LANE_COUNT)
      keys[i] = sort[t];

    uint16_t offsets[BIN_KEYS_PER_THREAD];

// WLMS
#pragma unroll
    for (unsigned int i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
      // CUB version "match any"
      /*
      unsigned warpFlags;
      #pragma unroll
      for (int k = 0; k < RADIX_LOG; ++k)
      {
          unsigned int mask;
          unsigned int current_bit = 1 << k + radixShift;
          asm("{\n"
              "    .reg .pred p;\n"
              "    and.b32 %0, %1, %2;"
              "    setp.ne.u32 p, %0, 0;\n"
              "    vote.ballot.sync.b32 %0, p, 0xffffffff;\n"
              "    @!p not.b32 %0, %0;\n"
              "}\n" : "=r"(mask) : "r"(keys[i]), "r"(current_bit));
          warpFlags = (k == 0) ? mask : warpFlags & mask;
      }
      const unsigned int bits = __popc(warpFlags & getLaneMaskLt());
      */
      unsigned warpFlags = 0xffffffff;
#pragma unroll
      for (int k = 0; k < RADIX_LOG; ++k) {
        const bool t2 = keys[i] >> k + radixShift & 1;
        warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
      }
      const unsigned int bits = __popc(warpFlags & getLaneMaskLt());

      // An alternative, but slightly slower version.
      /*
      offsets[i] = s_warpHist[keys[i] >> radixShift & RADIX_MASK] + bits;
      __syncwarp(0xffffffff);
      if (bits == 0)
          s_warpHist[keys[i] >> radixShift & RADIX_MASK] += __popc(warpFlags);
      __syncwarp(0xffffffff);
      */
      unsigned int preIncrementVal;
      if (bits == 0)
        preIncrementVal = atomicAdd(
            (unsigned int*)&s_warpHist[keys[i] >> radixShift & RADIX_MASK],
            __popc(warpFlags));

      offsets[i] =
          __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) + bits;
    }
    __syncthreads();

    // exclusive prefix sum up the warp histograms
    if (threadIdx.x < RADIX) {
      unsigned int reduction = s_warpHistograms[threadIdx.x];
      for (unsigned int i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE;
           i += RADIX) {
        reduction += s_warpHistograms[i];
        s_warpHistograms[i] = reduction - s_warpHistograms[i];
      }

      atomicAdd((unsigned int*)&passHistogram[threadIdx.x +
                                              (partitionIndex + 1) * RADIX],
                FLAG_REDUCTION | reduction << 2);

      // begin the exclusive prefix sum across the reductions
      s_localHistogram[threadIdx.x] = InclusiveWarpScanCircularShift(reduction);
    }
    __syncthreads();

    if (threadIdx.x < (RADIX >> LANE_LOG))
      s_localHistogram[threadIdx.x << LANE_LOG] =
          ActiveExclusiveWarpScan(s_localHistogram[threadIdx.x << LANE_LOG]);
    __syncthreads();

    if (threadIdx.x < RADIX && getLaneId())
      s_localHistogram[threadIdx.x] +=
          __shfl_sync(0xfffffffe, s_localHistogram[threadIdx.x - 1], 1);
    __syncthreads();

    // update offsets
    if (WARP_INDEX) {
#pragma unroll
      for (unsigned int i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
        const unsigned int t2 = keys[i] >> radixShift & RADIX_MASK;
        offsets[i] += s_warpHist[t2] + s_localHistogram[t2];
      }
    } else {
#pragma unroll
      for (unsigned int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        offsets[i] += s_localHistogram[keys[i] >> radixShift & RADIX_MASK];
    }
    __syncthreads();

// scatter keys into shared memory
#pragma unroll
    for (unsigned int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
      s_warpHistograms[offsets[i]] = keys[i];

    // split the warps into single thread cooperative groups and lookback
    if (threadIdx.x < RADIX) {
      unsigned int reduction = 0;
      for (unsigned int k = partitionIndex; k >= 0;) {
        const unsigned int flagPayload = passHistogram[threadIdx.x + k * RADIX];

        if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
          reduction += flagPayload >> 2;
          atomicAdd((unsigned int*)&passHistogram[threadIdx.x +
                                                  (partitionIndex + 1) * RADIX],
                    1 | (reduction << 2));
          s_localHistogram[threadIdx.x] =
              reduction - s_localHistogram[threadIdx.x];
          break;
        }

        if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION) {
          reduction += flagPayload >> 2;
          k--;
        }
      }
    }
    __syncthreads();

// scatter runs of keys into device memory
#pragma unroll
    for (unsigned int i = threadIdx.x; i < BIN_PART_SIZE; i += blockDim.x)
      alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] +
          i] = s_warpHistograms[i];
  }

  // Process the final partition slightly differently
  if (partitionIndex == gridDim.x - 1) {
    // immediately begin lookback
    if (threadIdx.x < RADIX) {
      if (partitionIndex) {
        unsigned int reduction = 0;
        for (unsigned int k = partitionIndex; k >= 0;) {
          const unsigned int flagPayload =
              passHistogram[threadIdx.x + k * RADIX];

          if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
            reduction += flagPayload >> 2;
            s_localHistogram[threadIdx.x] = reduction;
            break;
          }

          if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION) {
            reduction += flagPayload >> 2;
            k--;
          }
        }
      } else {
        s_localHistogram[threadIdx.x] = passHistogram[threadIdx.x] >> 2;
      }
    }
    __syncthreads();

    const unsigned int partEnd = BIN_PART_START + BIN_PART_SIZE;
    for (unsigned int i = threadIdx.x + BIN_PART_START; i < partEnd;
         i += blockDim.x) {
      unsigned int key;
      unsigned int offset;
      unsigned warpFlags = 0xffffffff;

      if (i < size) key = sort[i];

#pragma unroll
      for (unsigned int k = 0; k < RADIX_LOG; ++k) {
        const bool t = key >> k + radixShift & 1;
        warpFlags &= (t ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t);
      }
      const unsigned int bits = __popc(warpFlags & getLaneMaskLt());

#pragma unroll
      for (unsigned int k = 0; k < BIN_WARPS; ++k) {
        unsigned int preIncrementVal;
        if (WARP_INDEX == k && bits == 0 && i < size)
          preIncrementVal =
              atomicAdd(&s_localHistogram[key >> radixShift & RADIX_MASK],
                        __popc(warpFlags));

        if (WARP_INDEX == k)
          offset =
              __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) +
              bits;
        __syncthreads();
      }

      if (i < size) alt[offset] = key;
    }
  }
}

//
// ============================================================================
// Yanwen's version
//
// ============================================================================

__global__ void k_DigitBinningPass(unsigned int* sort,
                                   unsigned int* alt,
                                   volatile unsigned int* passHistogram,
                                   volatile unsigned int* index,
                                   unsigned int size,
                                   unsigned int radixShift) {
  __shared__ unsigned int s_warpHistograms[BIN_PART_SIZE];
  __shared__ unsigned int s_localHistogram[RADIX];

  volatile unsigned int* s_warpHist =
      &s_warpHistograms[WARP_INDEX << RADIX_LOG];

  const auto logicalBlocks = cub::DivideAndRoundUp(size, BIN_PART_SIZE);

  for (auto yanwen_block_id = blockIdx.x; yanwen_block_id < logicalBlocks;
       yanwen_block_id += gridDim.x) {
    // clear shared memory
    for (unsigned int i = threadIdx.x; i < BIN_HISTS_SIZE;
         i += blockDim.x)  // unnecessary work for last partion but still a win
                           // to avoid another barrier
      s_warpHistograms[i] = 0;

    // atomically assign partition tiles
    if (threadIdx.x == 0)
      s_warpHistograms[BIN_PART_SIZE - 1] =
          atomicAdd((unsigned int*)&index[radixShift >> 3], 1);
    __syncthreads();
    const unsigned int partitionIndex = s_warpHistograms[BIN_PART_SIZE - 1];

    // To handle input sizes not perfect multiples of the partition tile size
    if (partitionIndex < logicalBlocks - 1) {
      // load keys
      unsigned int keys[BIN_KEYS_PER_THREAD];
#pragma unroll
      for (unsigned int i = 0,
                        t = getLaneId() + BIN_SUB_PART_START + BIN_PART_START;
           i < BIN_KEYS_PER_THREAD;
           ++i, t += LANE_COUNT)
        keys[i] = sort[t];

      uint16_t offsets[BIN_KEYS_PER_THREAD];

// WLMS
#pragma unroll
      for (unsigned int i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
        // CUB version "match any"
        /*
        unsigned warpFlags;
        #pragma unroll
        for (int k = 0; k < RADIX_LOG; ++k)
        {
            unsigned int mask;
            unsigned int current_bit = 1 << k + radixShift;
            asm("{\n"
                "    .reg .pred p;\n"
                "    and.b32 %0, %1, %2;"
                "    setp.ne.u32 p, %0, 0;\n"
                "    vote.ballot.sync.b32 %0, p, 0xffffffff;\n"
                "    @!p not.b32 %0, %0;\n"
                "}\n" : "=r"(mask) : "r"(keys[i]), "r"(current_bit));
            warpFlags = (k == 0) ? mask : warpFlags & mask;
        }
        const unsigned int bits = __popc(warpFlags & getLaneMaskLt());
        */
        unsigned warpFlags = 0xffffffff;
#pragma unroll
        for (int k = 0; k < RADIX_LOG; ++k) {
          const bool t2 = keys[i] >> k + radixShift & 1;
          warpFlags &= (t2 ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t2);
        }
        const unsigned int bits = __popc(warpFlags & getLaneMaskLt());

        // An alternative, but slightly slower version.
        /*
        offsets[i] = s_warpHist[keys[i] >> radixShift & RADIX_MASK] + bits;
        __syncwarp(0xffffffff);
        if (bits == 0)
            s_warpHist[keys[i] >> radixShift & RADIX_MASK] += __popc(warpFlags);
        __syncwarp(0xffffffff);
        */
        unsigned int preIncrementVal;
        if (bits == 0)
          preIncrementVal = atomicAdd(
              (unsigned int*)&s_warpHist[keys[i] >> radixShift & RADIX_MASK],
              __popc(warpFlags));

        offsets[i] =
            __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) +
            bits;
      }
      __syncthreads();

      // exclusive prefix sum up the warp histograms
      if (threadIdx.x < RADIX) {
        unsigned int reduction = s_warpHistograms[threadIdx.x];
        for (unsigned int i = threadIdx.x + RADIX; i < BIN_HISTS_SIZE;
             i += RADIX) {
          reduction += s_warpHistograms[i];
          s_warpHistograms[i] = reduction - s_warpHistograms[i];
        }

        atomicAdd((unsigned int*)&passHistogram[threadIdx.x +
                                                (partitionIndex + 1) * RADIX],
                  FLAG_REDUCTION | reduction << 2);

        // begin the exclusive prefix sum across the reductions
        s_localHistogram[threadIdx.x] =
            InclusiveWarpScanCircularShift(reduction);
      }
      __syncthreads();

      if (threadIdx.x < (RADIX >> LANE_LOG))
        s_localHistogram[threadIdx.x << LANE_LOG] =
            ActiveExclusiveWarpScan(s_localHistogram[threadIdx.x << LANE_LOG]);
      __syncthreads();

      if (threadIdx.x < RADIX && getLaneId())
        s_localHistogram[threadIdx.x] +=
            __shfl_sync(0xfffffffe, s_localHistogram[threadIdx.x - 1], 1);
      __syncthreads();

      // update offsets
      if (WARP_INDEX) {
#pragma unroll
        for (unsigned int i = 0; i < BIN_KEYS_PER_THREAD; ++i) {
          const unsigned int t2 = keys[i] >> radixShift & RADIX_MASK;
          offsets[i] += s_warpHist[t2] + s_localHistogram[t2];
        }
      } else {
#pragma unroll
        for (unsigned int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
          offsets[i] += s_localHistogram[keys[i] >> radixShift & RADIX_MASK];
      }
      __syncthreads();

// scatter keys into shared memory
#pragma unroll
      for (unsigned int i = 0; i < BIN_KEYS_PER_THREAD; ++i)
        s_warpHistograms[offsets[i]] = keys[i];

      // split the warps into single thread cooperative groups and lookback
      if (threadIdx.x < RADIX) {
        unsigned int reduction = 0;
        for (unsigned int k = partitionIndex; k >= 0;) {
          const unsigned int flagPayload =
              passHistogram[threadIdx.x + k * RADIX];

          if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
            reduction += flagPayload >> 2;
            atomicAdd(
                (unsigned int*)&passHistogram[threadIdx.x +
                                              (partitionIndex + 1) * RADIX],
                1 | (reduction << 2));
            s_localHistogram[threadIdx.x] =
                reduction - s_localHistogram[threadIdx.x];
            break;
          }

          if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION) {
            reduction += flagPayload >> 2;
            k--;
          }
        }
      }
      __syncthreads();

// scatter runs of keys into device memory
#pragma unroll
      for (unsigned int i = threadIdx.x; i < BIN_PART_SIZE; i += blockDim.x)
        alt[s_localHistogram[s_warpHistograms[i] >> radixShift & RADIX_MASK] +
            i] = s_warpHistograms[i];
    }

    // Process the final partition slightly differently
    if (partitionIndex == logicalBlocks - 1) {
      // immediately begin lookback
      if (threadIdx.x < RADIX) {
        if (partitionIndex) {
          unsigned int reduction = 0;
          for (unsigned int k = partitionIndex; k >= 0;) {
            const unsigned int flagPayload =
                passHistogram[threadIdx.x + k * RADIX];

            if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
              reduction += flagPayload >> 2;
              s_localHistogram[threadIdx.x] = reduction;
              break;
            }

            if ((flagPayload & FLAG_MASK) == FLAG_REDUCTION) {
              reduction += flagPayload >> 2;
              k--;
            }
          }
        } else {
          s_localHistogram[threadIdx.x] = passHistogram[threadIdx.x] >> 2;
        }
      }
      __syncthreads();

      const unsigned int partEnd = BIN_PART_START + BIN_PART_SIZE;
      for (unsigned int i = threadIdx.x + BIN_PART_START; i < partEnd;
           i += blockDim.x) {
        unsigned int key;
        unsigned int offset;
        unsigned warpFlags = 0xffffffff;

        if (i < size) key = sort[i];

#pragma unroll
        for (unsigned int k = 0; k < RADIX_LOG; ++k) {
          const bool t = key >> k + radixShift & 1;
          warpFlags &= (t ? 0 : 0xffffffff) ^ __ballot_sync(0xffffffff, t);
        }
        const unsigned int bits = __popc(warpFlags & getLaneMaskLt());

#pragma unroll
        for (unsigned int k = 0; k < BIN_WARPS; ++k) {
          unsigned int preIncrementVal;
          if (WARP_INDEX == k && bits == 0 && i < size)
            preIncrementVal =
                atomicAdd(&s_localHistogram[key >> radixShift & RADIX_MASK],
                          __popc(warpFlags));

          if (WARP_INDEX == k)
            offset =
                __shfl_sync(0xffffffff, preIncrementVal, __ffs(warpFlags) - 1) +
                bits;
          __syncthreads();
        }

        if (i < size) alt[offset] = key;
      }
    }
  }
}

}  // namespace gpu

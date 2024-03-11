#pragma once

#include "handlers/pipe.h"

namespace gpu::v2 {

void dispatch_ComputeMorton(int grid_size, cudaStream_t stream, Pipe& pipe);

void dispatch_RadixSort(int grid_size, cudaStream_t stream, const Pipe& pipe);

void dispatch_RemoveDuplicates(int grid_size, cudaStream_t stream, Pipe& pipe);

void dispatch_BuildRadixTree(int grid_size,
                             cudaStream_t stream,
                             const Pipe& pipe);

void dispatch_EdgeCount(int grid_size, cudaStream_t stream, const Pipe& pipe);

void dispatch_EdgeOffset(int grid_size, cudaStream_t stream, const Pipe& pipe);

void dispatch_BuildOctree(int grid_size, cudaStream_t stream, const Pipe& pipe);

}  // namespace gpu::v2

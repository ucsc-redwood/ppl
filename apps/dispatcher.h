#pragma once

#include "handlers/pipe.cuh"

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

namespace cpu::v2 {

void dispatch_ComputeMorton(int n_threads, const Pipe& pipe);

void dispatch_RadixSort(int n_threads, const Pipe& pipe);

void dispatch_RemoveDuplicates(int n_threads, Pipe& pipe);

void dispatch_BuildRadixTree(int n_threads, const Pipe& pipe);

void dispatch_EdgeCount(int n_threads, const Pipe& pipe);

void dispatch_EdgeOffset(int n_threads, Pipe& pipe);

void dispatch_BuildOctree(int n_threads, const Pipe& pipe);

}  // namespace cpu::v2

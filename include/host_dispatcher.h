#pragma once

#include "handlers/pipe.h"

namespace cpu::v2 {

void dispatch_ComputeMorton(int n_threads, const Pipe& pipe);

void dispatch_RadixSort(int n_threads, const Pipe& pipe);

void dispatch_RemoveDuplicates(int n_threads, Pipe& pipe);

void dispatch_BuildRadixTree(int n_threads, const Pipe& pipe);

void dispatch_EdgeCount(int n_threads, const Pipe& pipe);

void dispatch_EdgeOffset(int n_threads, Pipe& pipe);

void dispatch_BuildOctree(int n_threads, const Pipe& pipe);

}  // namespace cpu::v2

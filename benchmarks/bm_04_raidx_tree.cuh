#pragma once

#include "cu_bench_helper.cuh"

void BM_GPU_RadixTree(bm::State& st);
void BM_CPU_RadixTree(bm::State& st);

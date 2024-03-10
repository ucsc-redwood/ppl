#pragma once

#include "cu_bench_helper.cuh"

void BM_GPU_PrefixSum(bm::State& st);
void BM_CPU_PrefixSum(bm::State& st);

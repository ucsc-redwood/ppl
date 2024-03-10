#pragma once

#include "cu_bench_helper.cuh"

void BM_GPU_Sort(bm::State& st);
void BM_CPU_Sort(bm::State& st);
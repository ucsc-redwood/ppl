#pragma once

// Device
#include "cuda/agents/prefix_sum_agent.cuh"
#include "cuda/agents/unique_agent.cuh"
#include "cuda/kernels/01_morton.cuh"
#include "cuda/kernels/02_sort.cuh"
#include "cuda/kernels/03_unique.cuh"
#include "cuda/kernels/04_radix_tree.cuh"
#include "cuda/kernels/05_edge_count.cuh"
#include "cuda/kernels/06_prefix_sum.cuh"
#include "cuda/kernels/07_octree.cuh"

// Host
#include "openmp/kernels/00_init.hpp"
#include "openmp/kernels/01_morton.hpp"
#include "openmp/kernels/02_sort.hpp"
#include "openmp/kernels/03_unique.hpp"
#include "openmp/kernels/04_radix_tree.hpp"
#include "openmp/kernels/05_edge_count.hpp"
#include "openmp/kernels/06_prefix_sum.hpp"
#include "openmp/kernels/07_octree.hpp"

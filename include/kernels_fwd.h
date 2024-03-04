#pragma once

// Device
#include "cuda/agents/prefix_sum_agent.cuh"
#include "cuda/dispatchers/edge_count_dispatch.cuh"
#include "cuda/dispatchers/init_dispatch.cuh"
#include "cuda/dispatchers/morton_dispatch.cuh"
#include "cuda/dispatchers/octree_dispatch.cuh"
#include "cuda/dispatchers/prefix_sum_dispatch.cuh"
#include "cuda/dispatchers/radix_tree_dispatch.cuh"
#include "cuda/dispatchers/sort_dispatch.cuh"
#include "cuda/dispatchers/unique_dispatch.cuh"

// Host
#include "openmp/kernels/00_init.hpp"
#include "openmp/kernels/01_morton.hpp"
#include "openmp/kernels/02_sort.hpp"
#include "openmp/kernels/03_unique.hpp"
#include "openmp/kernels/04_radix_tree.hpp"
#include "openmp/kernels/05_edge_count.hpp"
#include "openmp/kernels/06_prefix_sum.hpp"
#include "openmp/kernels/07_octree.hpp"

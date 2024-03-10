#pragma once

#include <numeric>

namespace cpu {

inline void std_exclusive_scan(const int* u_input, int* u_output, int n) {
  std::exclusive_scan(u_input, u_input + n, u_output, 0);
}

// template <typename T>
// __global__ void k_PrefixSumLocal(const T* u_input,
//                                  T* u_output,
//                                  int n,
//                                  T* u_auxiliary = nullptr);

// template <typename T>
// __global__ void k_SingleBlockExclusiveScan(const T* u_input,
//                                            T* u_output,
//                                            int n);

// template <typename T>
// __global__ void k_MakeGlobalPrefixSum(const T* u_local_sums,
//                                       const T* u_auxiliary_summed,
//                                       T* u_global_sums,
//                                       int n);

}  // namespace cpu

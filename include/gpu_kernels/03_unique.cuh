#pragma once

namespace gpu {

template <typename T>
__global__ void k_BetterFindDups(const T *u_keys, int *u_flag_heads, int n);

template <typename T>
__global__ void k_MoveDups(const T *u_keys,
                           const int *u_flag_heads_sums,
                           const int N /*old n*/,
                           T *u_keys_out,
                           int *n_unique_out);

}  // namespace gpu
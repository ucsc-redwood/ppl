#pragma once

namespace gpu {

template <typename T>
__global__ void k_LocalPrefixSums(const T *u_input, T *u_output, const int n);

}  // namespace gpu

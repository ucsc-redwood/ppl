#include <cub/cub.cuh>

#include "cuda/agents/unique_agent.cuh"
#include "cuda/kernels/03_unique.cuh"

namespace gpu {

__global__ void k_FindDups(const unsigned int *u_keys,
                           int *u_flag_heads,
                           const int n) {
  __shared__ UniqueAgent::TempStorage temp_storage;

  UniqueAgent agent(n);
  agent.Process_FindDups(temp_storage, u_keys, u_flag_heads, n);
}

__global__ void k_MoveDups(const unsigned int *u_keys,
                           const int *u_flag_heads_sums,
                           const int n,
                           unsigned int *u_keys_out,
                           int *n_unique_out) {
  UniqueAgent agent(n);
  agent.Process_MoveDups(
      u_keys, u_flag_heads_sums, n, u_keys_out, n_unique_out);
}

}  // namespace gpu

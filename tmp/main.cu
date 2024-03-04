#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "cuda/agents/expr_cub_scan_agent.cuh"
#include "cuda/helper.cuh"
#include "cuda/unified_vector.cuh"

template <typename ChainedPolicyT, typename ScanTileStateT>
__global__ void Hey(const int* d_in,
                    int* d_out,
                    ScanTileStateT tile_state,
                    int start_tile,
                    int num_items) {
  using ScanPolicyT = ChainedPolicyT::ActivePolicy::ScanPolicyT;
  // typedef AgentScan<ScanPolicyT,
  // InputIteratorT,
  // OutputIteratorT,
  // ScanOpT,
  // RealInitValueT,
  // OffsetT,
  // AccumT>
  using AgentScanT =
      cub::AgentScan<ScanPolicyT, int, int, cub::Sum(), int, int, int>;

  __shared__ typename AgentScanT::TempStorage temp_storage;

  auto scan_op = cub::Sum();
  auto real_init_value = 0;
  AgentScanT(temp_storage, d_in, d_out, scan_op, real_init_value)
      .ConsumeRange(num_items, tile_state, start_tile);
}

int main() {
  constexpr int n = 2048;

  constexpr int n_threads = gpu::expr::ScanAgent<int>::n_threads;
  constexpr int tile_size = gpu::expr::ScanAgent<int>::tile_size;

  using ScanTileStateT = cub::ScanTileState<int>;

  const cu::unified_vector<int> u_data(n, 1);
  cu::unified_vector<int> u_output(n);

  Hey<<<1, n_threads>>>(u_data.data(), u_output.data(), n);
  SYNC_DEVICE();

  return 0;
}
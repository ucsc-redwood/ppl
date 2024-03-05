#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "cuda/agents/expr_cub_scan_agent.cuh"
#include "cuda/helper.cuh"
#include "cuda/unified_vector.cuh"


using MyAgentScanT = gpu::expr::AgentScan<cub::Sum, int, int, int>;

template <typename ScanTileStateT>
__global__ void Hey(int* d_in,
                    int* d_out,
                    ScanTileStateT tile_state,
                    int start_tile,
                    int num_items) {
  __shared__ MyAgentScanT::TempStorage temp_storage;

  auto scan_op = cub::Sum();
  auto real_init_value = 0;
  //MyAgentScanT(temp_storage, d_in, d_out, scan_op, real_init_value)
  //    .ConsumeRange(num_items, tile_state, start_tile);

  auto agent = MyAgentScanT(temp_storage,
                            d_in,
                            d_out,
                            scan_op,
                            real_init_value);
  //agent.ConsumeRange(0, num_items);
  agent.ConsumeRange(0, num_items);
}

int main() {
  constexpr int n = 2048;

  constexpr int n_threads = MyAgentScanT::n_threads;
  constexpr int tile_size = MyAgentScanT::TILE_ITEMS;

  using ScanTileStateT = cub::ScanTileState<int>;
  ScanTileStateT tile_state;

  cu::unified_vector<int> u_data(n, 1);
  cu::unified_vector<int> u_output(n);

  Hey<<<1, n_threads>>>(u_data.data(), u_output.data(), tile_state, 0, n);
  SYNC_DEVICE();

  for (int i = 0; i < n; i++) {
    std::cout << i << ":\t" << u_output[i] << "\n";
  }

  return 0;
}
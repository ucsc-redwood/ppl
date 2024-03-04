#pragma once

#include <device_launch_parameters.h>

#include <cub/cub.cuh>

namespace gpu {

namespace expr {

// I am only using this as PrefixSum,
template <typename T>
struct ScanAgent {
  using ScanOpT = cub::Sum;

  using AccumT = T;
  using ScanTileStateT = cub::ScanTileState<AccumT>;

  static constexpr auto n_threads = 128;
  static constexpr auto items_per_thread = 8;
  static constexpr auto tile_size = n_threads * items_per_thread;

  using BlockLoad = cub::
      BlockLoad<T, n_threads, items_per_thread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockScan = cub::BlockScan<T, n_threads>;
  using BlockStore = cub::BlockStore<T,
                                     n_threads,
                                     items_per_thread,
                                     cub::BLOCK_STORE_WARP_TRANSPOSE>;

  // Callback type for obtaining tile prefix during block scan
  //   using DelayConstructorT =
  //       typename cub::AgentScanPolicyT::detail::delay_constructor_t;
  using TilePrefixCallbackOpT =
      cub::TilePrefixCallbackOp<AccumT, ScanOpT, ScanTileStateT, 0 /* PTX */>;

//   union {
//     typename BlockLoad::TempStorage load;
//     typename BlockStore::TempStorage store;

//     struct ScanStorage {
//       typename TilePrefixCallbackOpT::TempStorage prefix;
//       typename BlockScan::TempStorage scan;
//     } scan_storage;
//   };

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  TempStorage& temp_storage;
  T* d_in;          // Input data
  T* d_out;         // Output data
  ScanOpT scan_op;  // Scan operator (e.g., cub::Sum)
  T init_value;     // Initial value for the scan, e.g. 0 for sum

  //---------------------------------------------------------------------
  // Block scan utility methods
  //---------------------------------------------------------------------

  /**
   * Exclusive scan specialization (first tile)
   */
  __device__ __forceinline__ void ScanTile(T (&items)[items_per_thread],
                                           T init_value,
                                           ScanOpT scan_op,
                                           T& block_aggregate) {
    BlockScan(temp_storage.scan_storage.scan)
        .ExclusiveScan(items, items, init_value);
    block_aggregate = scan_op(init_value, block_aggregate);
  }

  /**
   * Exclusive scan specialization (subsequent tiles)
   */
  template <typename PrefixCallback>
  __device__ __forceinline__ void ScanTile(T (&items)[items_per_thread],
                                           T init_value,
                                           PrefixCallback& prefix_op) {
    BlockScan(temp_storage.scan_storage.scan)
        .ExclusiveScan(items, items, init_value, prefix_op);
  }

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  __device__ __forceinline__ explicit ScanAgent(TempStorage& temp_storage,
                                                T* d_in,
                                                T* d_out,
                                                ScanOpT scan_op,
                                                T init_value)
      : temp_storage(temp_storage),
        d_in(d_in),
        d_out(d_out),
        scan_op(scan_op),
        init_value(init_value) {}

  //---------------------------------------------------------------------
  // Cooperatively scan a device-wide sequence of tiles with other CTAs
  //---------------------------------------------------------------------

  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void ConsumeTile(int num_remaining,
                                              int tile_idx,
                                              int tile_offset,
                                              ScanTileStateT& tile_state) {
    // Load items
    T items[items_per_thread];

    if (IS_LAST_TILE) {
      // Fill last element with the first element because collectives are
      // not suffix guarded.
      BlockLoad(temp_storage.load)
          .Load(
              d_in + tile_offset, items, num_remaining, *(d_in + tile_offset));

    } else {
      BlockLoad(temp_storage.load).Load(d_in + tile_offset, items);
    }

    __syncthreads();

    // Perform tile scan
    if (tile_idx == 0) {
      // Scan first tile
      AccumT block_aggregate;
      ScanTile(items, init_value, scan_op, block_aggregate);

      if ((!IS_LAST_TILE) && (threadIdx.x == 0)) {
        tile_state.SetInclusive(0, block_aggregate);
      }
    } else {
      // Scan non-first tile
      TilePrefixCallbackOpT prefix_op(
          tile_state, temp_storage.scan_storage.prefix, scan_op, tile_idx);
      ScanTile(items, scan_op, prefix_op);
    }

    __syncthreads();

    if (IS_LAST_TILE) {
      BlockStore(temp_storage.store)
          .Store(d_out + tile_offset, items, num_remaining);
    } else {
      BlockStore(temp_storage.store).Store(d_out + tile_offset, items);
    }
  }
};

}  // namespace expr

}  // namespace gpu
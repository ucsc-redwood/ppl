#pragma once

#include <device_launch_parameters.h>

#include <cub/cub.cuh>

namespace gpu {
namespace expr {

#define CTA_SYNC() __syncthreads()

// typename AgentScanPolicyT,
template <
    // typename InputIteratorT,
    // typename OutputIteratorT,
    typename ScanOpT,
    typename InitValueT,
    typename OffsetT,
    typename AccumT>

struct AgentScan {
  using InputIteratorT = int*;
  using OutputIteratorT = int*;

  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  // Tile status descriptor interface type
  using ScanTileStateT = cub::ScanTileState<AccumT>;

  static constexpr auto n_threads = 128;
  static constexpr auto ITEMS_PER_THREAD = 8;
  static constexpr auto TILE_ITEMS = n_threads * ITEMS_PER_THREAD;

  using BlockLoadT = cub::BlockLoad<AccumT, n_threads, ITEMS_PER_THREAD>;
  using BlockStoreT = cub::BlockStore<AccumT, n_threads, ITEMS_PER_THREAD>;
  using BlockScanT = cub::BlockScan<AccumT, ITEMS_PER_THREAD>;

  using TilePrefixCallbackOpT =
      cub::TilePrefixCallbackOp<AccumT, ScanOpT, ScanTileStateT>;

  // Stateful BlockScan prefix callback type for managing a running total while
  // scanning consecutive tiles
  using RunningPrefixCallbackOp =
      cub::BlockScanRunningPrefixOp<AccumT, ScanOpT>;

  // Shared memory type for this thread block
  union _TempStorage {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;

    struct ScanStorage {
      typename TilePrefixCallbackOpT::TempStorage prefix;
      typename BlockScanT::TempStorage scan;
    } scan_storage;
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : cub::Uninitialized<_TempStorage> {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  _TempStorage& temp_storage;  ///< Reference to temp_storage
  InputIteratorT d_in;         ///< Input data
  OutputIteratorT d_out;       ///< Output data
  ScanOpT scan_op;             ///< Binary scan operator
  InitValueT init_value;       ///< The init_value element for ScanOpT

  //---------------------------------------------------------------------
  // Block scan utility methods
  //---------------------------------------------------------------------

  /**
   * Exclusive scan specialization (first tile)
   */
  __device__ __forceinline__ void ScanTile(AccumT (&items)[ITEMS_PER_THREAD],
                                           AccumT init_value,
                                           ScanOpT scan_op,
                                           AccumT& block_aggregate) {
    BlockScanT(temp_storage.scan_storage.scan)
        .ExclusiveScan(items, items, init_value, scan_op, block_aggregate);
    block_aggregate = scan_op(init_value, block_aggregate);
  }

  /**
   * Exclusive scan specialization (subsequent tiles)
   */
  template <typename PrefixCallback>
  __device__ __forceinline__ void ScanTile(AccumT (&items)[ITEMS_PER_THREAD],
                                           ScanOpT scan_op,
                                           PrefixCallback& prefix_op) {
    BlockScanT(temp_storage.scan_storage.scan)
        .ExclusiveScan(items, items, scan_op, prefix_op);
  }

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  /**
   * @param temp_storage
   *   Reference to temp_storage
   *
   * @param d_in
   *   Input data
   *
   * @param d_out
   *   Output data
   *
   * @param scan_op
   *   Binary scan operator
   *
   * @param init_value
   *   Initial value to seed the exclusive scan
   */
  __device__ __forceinline__ AgentScan(TempStorage& temp_storage,
                                       InputIteratorT d_in,
                                       OutputIteratorT d_out,
                                       ScanOpT scan_op,
                                       InitValueT init_value)
      : temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_out(d_out),
        scan_op(scan_op),
        init_value(init_value) {}

  //---------------------------------------------------------------------
  // Cooperatively scan a device-wide sequence of tiles with other CTAs
  //---------------------------------------------------------------------

  template <bool IS_LAST_TILE>
  __device__ __forceinline__ void ConsumeTile(OffsetT num_remaining,
                                              int tile_idx,
                                              OffsetT tile_offset,
                                              ScanTileStateT& tile_state) {
    // Load items
    AccumT items[ITEMS_PER_THREAD];

    if (IS_LAST_TILE) {
      // Fill last element with the first element because collectives are
      // not suffix guarded.
      BlockLoadT(temp_storage.load)
          .Load(
              d_in + tile_offset, items, num_remaining, *(d_in + tile_offset));
    } else {
      BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items);
    }

    CTA_SYNC();

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

    CTA_SYNC();

    // Store items
    if (IS_LAST_TILE) {
      BlockStoreT(temp_storage.store)
          .Store(d_out + tile_offset, items, num_remaining);
    } else {
      BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items);
    }
  }

  /**
   * @brief Scan tiles of items as part of a dynamic chained scan
   *
   * @param num_items
   *   Total number of input items
   *
   * @param tile_state
   *   Global tile state descriptor
   *
   * @param start_tile
   *   The starting tile for the current grid
   */
  __device__ __forceinline__ void ConsumeRange(OffsetT num_items,
                                               ScanTileStateT& tile_state,
                                               int start_tile) {
    // Blocks are launched in increasing order, so just assign one tile per
    // block

    // Current tile index
    int tile_idx = start_tile + blockIdx.x;

    // Global offset for the current tile
    OffsetT tile_offset = OffsetT(TILE_ITEMS) * tile_idx;

    // Remaining items (including this tile)
    OffsetT num_remaining = num_items - tile_offset;

    if (num_remaining > TILE_ITEMS) {
      // Not last tile
      ConsumeTile<false>(num_remaining, tile_idx, tile_offset, tile_state);
    } else if (num_remaining > 0) {
      // Last tile
      ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state);
    }
  }

  //---------------------------------------------------------------------------
  // Scan an sequence of consecutive tiles (independent of other thread blocks)
  //---------------------------------------------------------------------------

  /**
   * @brief Process a tile of input
   *
   * @param tile_offset
   *   Tile offset
   *
   * @param prefix_op
   *   Running prefix operator
   *
   * @param valid_items
   *   Number of valid items in the tile
   */
  template <bool IS_FIRST_TILE, bool IS_LAST_TILE>
  __device__ __forceinline__ void ConsumeTile(
      OffsetT tile_offset,
      RunningPrefixCallbackOp& prefix_op,
      int valid_items = TILE_ITEMS) {
    // Load items
    AccumT items[ITEMS_PER_THREAD];

    if (IS_LAST_TILE) {
      // Fill last element with the first element because collectives are
      // not suffix guarded.
      BlockLoadT(temp_storage.load)
          .Load(d_in + tile_offset, items, valid_items, *(d_in + tile_offset));
    } else {
      BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items);
    }

    CTA_SYNC();

    // Block scan
    if (IS_FIRST_TILE) {
      AccumT block_aggregate;
      ScanTile(items, init_value, scan_op, block_aggregate);
      prefix_op.running_total = block_aggregate;
    } else {
      ScanTile(items, scan_op, prefix_op);
    }

    CTA_SYNC();

    // Store items
    if (IS_LAST_TILE) {
      BlockStoreT(temp_storage.store)
          .Store(d_out + tile_offset, items, valid_items);
    } else {
      BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items);
    }
  }

  /**
   * @brief Scan a consecutive share of input tiles
   *
   * @param[in] range_offset
   *   Threadblock begin offset (inclusive)
   *
   * @param[in] range_end
   *   Threadblock end offset (exclusive)
   */
  __device__ __forceinline__ void ConsumeRange(OffsetT range_offset,
                                               OffsetT range_end) {
    cub::BlockScanRunningPrefixOp<AccumT, ScanOpT> prefix_op(scan_op);

    if (range_offset + TILE_ITEMS <= range_end) {
      // Consume first tile of input (full)
      ConsumeTile<true, true>(range_offset, prefix_op);
      range_offset += TILE_ITEMS;

      // Consume subsequent full tiles of input
      while (range_offset + TILE_ITEMS <= range_end) {
        ConsumeTile<false, true>(range_offset, prefix_op);
        range_offset += TILE_ITEMS;
      }

      // Consume a partially-full tile
      if (range_offset < range_end) {
        int valid_items = range_end - range_offset;
        ConsumeTile<false, false>(range_offset, prefix_op, valid_items);
      }
    } else {
      // Consume the first tile of input (partially-full)
      int valid_items = range_end - range_offset;
      ConsumeTile<true, false>(range_offset, prefix_op, valid_items);
    }
  }

  /**
   * @brief Scan a consecutive share of input tiles, seeded with the
   *        specified prefix value
   * @param[in] range_offset
   *   Threadblock begin offset (inclusive)
   *
   * @param[in] range_end
   *   Threadblock end offset (exclusive)
   *
   * @param[in] prefix
   *   The prefix to apply to the scan segment
   */
  __device__ __forceinline__ void ConsumeRange(OffsetT range_offset,
                                               OffsetT range_end,
                                               AccumT prefix) {
    cub::BlockScanRunningPrefixOp<AccumT, ScanOpT> prefix_op(prefix, scan_op);

    // Consume full tiles of input
    while (range_offset + TILE_ITEMS <= range_end) {
      ConsumeTile<true, false>(range_offset, prefix_op);
      range_offset += TILE_ITEMS;
    }

    // Consume a partially-full tile
    if (range_offset < range_end) {
      int valid_items = range_end - range_offset;
      ConsumeTile<false, false>(range_offset, prefix_op, valid_items);
    }
  }
};

//// I am only using this as PrefixSum,
//
// template <typename T>
// struct ScanAgent {
//  using ScanOpT = cub::Sum;
//
//  using AccumT = T;
//  using ScanTileStateT = cub::ScanTileState<AccumT>;
//
//  static constexpr auto n_threads = 128;
//  static constexpr auto ITEMS_PER_THREAD = 8;
//  static constexpr auto tile_size = n_threads * ITEMS_PER_THREAD;
//
//  using BlockLoad = cub::BlockLoad<
//    T, n_threads, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
//  using BlockScan = cub::BlockScan<T, n_threads>;
//  using BlockStore = cub::BlockStore<T,
//                                     n_threads,
//                                     ITEMS_PER_THREAD,
//                                     cub::BLOCK_STORE_WARP_TRANSPOSE>;
//
//
//  using TilePrefixCallbackOpT = cub::TilePrefixCallbackOp<
//    AccumT, ScanOpT, ScanTileStateT>;
//
//  union _TempStorage {
//    typename BlockLoad::TempStorage load;
//    typename BlockStore::TempStorage store;
//
//    struct ScanStorage {
//      typename TilePrefixCallbackOpT::TempStorage prefix;
//      typename BlockScan::TempStorage scan;
//    } scan_storage;
//  };
//
//  struct TempStorage : cub::Uninitialized<_TempStorage> {
//  };
//
//
//  //---------------------------------------------------------------------
//  // Per-thread fields
//  //---------------------------------------------------------------------
//
//  TempStorage& temp_storage;
//  T* d_in;         // Input data
//  T* d_out;        // Output data
//  ScanOpT scan_op; // Scan operator (e.g., cub::Sum)
//  T init_value;    // Initial value for the scan, e.g. 0 for sum
//
//  //---------------------------------------------------------------------
//  // Block scan utility methods
//  //---------------------------------------------------------------------
//
//  /**
//   * Exclusive scan specialization (first tile)
//   */
//  __device__ __forceinline__ void ScanTile(T (&items)[ITEMS_PER_THREAD],
//                                           T init_value,
//                                           ScanOpT scan_op,
//                                           T& block_aggregate) {
//    BlockScan(temp_storage.scan_storage.scan)
//        .ExclusiveScan(items, items, init_value);
//    block_aggregate = scan_op(init_value, block_aggregate);
//  }
//
//  /**
//   * Exclusive scan specialization (subsequent tiles)
//   */
//  template <typename PrefixCallback>
//  __device__ __forceinline__ void ScanTile(T (&items)[ITEMS_PER_THREAD],
//                                           T init_value,
//                                           PrefixCallback& prefix_op) {
//    BlockScan(temp_storage.scan_storage.scan)
//        .ExclusiveScan(items, items, init_value, prefix_op);
//  }
//
//  //---------------------------------------------------------------------
//  // Constructor
//  //---------------------------------------------------------------------
//
//  __device__ __forceinline__ explicit ScanAgent(TempStorage& temp_storage,
//                                                T* d_in,
//                                                T* d_out,
//                                                ScanOpT scan_op,
//                                                T init_value)
//    : temp_storage(temp_storage.Alias()),
//      d_in(d_in),
//      d_out(d_out),
//      scan_op(scan_op),
//      init_value(init_value) {
//  }
//
//  //---------------------------------------------------------------------
//  // Cooperatively scan a device-wide sequence of tiles with other CTAs
//  //---------------------------------------------------------------------
//
//  template <bool IS_LAST_TILE>
//  __device__ __forceinline__ void ConsumeTile(int num_remaining,
//                                              int tile_idx,
//                                              int tile_offset,
//                                              ScanTileStateT& tile_state) {
//    // Load items
//    T items[ITEMS_PER_THREAD];
//
//    if (IS_LAST_TILE) {
//      // Fill last element with the first element because collectives are
//      // not suffix guarded.
//      BlockLoad(temp_storage.load)
//          .Load(
//              d_in + tile_offset,
//              items,
//              num_remaining,
//              *(d_in + tile_offset));
//    } else {
//      BlockLoad(temp_storage.load).Load(d_in + tile_offset, items);
//    }
//
//    __syncthreads();
//
//    // Perform tile scan
//    if (tile_idx == 0) {
//      // Scan first tile
//      AccumT block_aggregate;
//      ScanTile(items, init_value, scan_op, block_aggregate);
//
//      if ((!IS_LAST_TILE) && (threadIdx.x == 0)) {
//        tile_state.SetInclusive(0, block_aggregate);
//      }
//    } else {
//      // Scan non-first tile
//      TilePrefixCallbackOpT prefix_op(
//          tile_state,
//          temp_storage.scan_storage.prefix,
//          scan_op,
//          tile_idx);
//      ScanTile(items, scan_op, prefix_op);
//    }
//
//    __syncthreads();
//
//    if (IS_LAST_TILE) {
//      BlockStore(temp_storage.store)
//          .Store(d_out + tile_offset, items, num_remaining);
//    } else {
//      BlockStore(temp_storage.store).Store(d_out + tile_offset, items);
//    }
//  }
//
//  __device__ __forceinline__ void ConsumeRange(int num_items,
//                                               ScanTileStateT& tile_state,
//                                               int start_tile) {
//    // Blocks are launched in increasing order, so just assign one tile per
//    // block
//
//    // Current tile index
//    int tile_idx = start_tile + blockIdx.x;
//
//    // Global offset for the current tile
//    auto tile_offset = tile_idx * tile_size;
//
//    // Remaining items (including this tile)
//    auto num_remaining = num_items - tile_offset;
//
//    if (num_remaining > tile_size) {
//      // Not last tile
//      ConsumeTile<false>(num_remaining, tile_idx, tile_offset, tile_state);
//    } else {
//      // Last tile
//      ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state);
//    }
//  }
//};
}  // namespace expr
}  // namespace gpu
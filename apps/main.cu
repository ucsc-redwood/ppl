#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>

#include "app_params.hpp"
#include "types/pipe.cuh"

//
#include "cuda/dispatchers/init_dispatch.cuh"
#include "cuda/dispatchers/morton_dispatch.cuh"

// ok let's list out some memories we need

// Essential:
// 1. u_points (n)
// 2. u_morton_keys (n)
//    - u_sort_alt
//    - u_global_histogram
//    - u_index
//    - u_pass_histogram (x4)
// 3. u_unique_keys (n_unique)
//    - u_flag_heads
//    - u_aux (optional)
// 4. Radix Node data...
// 5. u_edge_count
// 6. u_edge_offset
//     - u_auxlitary
// 7. Octree Node data...

struct NaivePipe {
  int n;

  cu::unified_vector<glm::vec4> u_points;
  cu::unified_vector<unsigned int> u_morton_keys;

  struct {
    cu::unified_vector<unsigned int> u_sort_alt;
    cu::unified_vector<unsigned int> u_global_histogram;
    cu::unified_vector<unsigned int> u_index;
    std::array<cu::unified_vector<unsigned int>, 4> u_pass_histogram;
  } sort_tmp;

 public:
  explicit NaivePipe(const int n) : n(n) {
    u_points.resize(n);
    u_morton_keys.resize(n);

    sort_tmp.u_sort_alt.resize(n);
    sort_tmp.u_global_histogram.resize(256 * 4);
    sort_tmp.u_index.resize(4);
    // for (auto& pass : sort_tmp.u_pass_histogram) {
    // pass.resize(256 *
    // }
  }

  void attachStream(const cudaStream_t stream) {
    ATTACH_STREAM_SINGLE(u_points.data());
    ATTACH_STREAM_SINGLE(u_morton_keys.data());
    ATTACH_STREAM_SINGLE(sort_tmp.u_sort_alt.data());
    ATTACH_STREAM_SINGLE(sort_tmp.u_global_histogram.data());
    ATTACH_STREAM_SINGLE(sort_tmp.u_index.data());
    for (auto& pass : sort_tmp.u_pass_histogram) {
      ATTACH_STREAM_SINGLE(pass.data());
    }
  }

}

void new_dispatch_Sort(
  const int grid_size,
  const cudaStream_t stream,

  unsigned int* u_sort,
  unsigned int* u_sort_alt,
  unsigned int* u_global_histogram,
  unsigned int* u_index,
  unsigned int* u_pass_histogram,
  size_t n
){
  const auto logical_blocks = cub::DivideAndRoundUp(n, 65536);

  gpu::k_GlobalHistogram_WithLogicalBlocks(
      u_sort, u_global_histogram, n, logical_blocks
      // grid_size,
  )
}

int main(const int argc, const char** argv) {
  AppParams params(argc, argv);
  params.print_params();

  spdlog::set_level(params.debug_print ? spdlog::level::debug
                                       : spdlog::level::info);

  // ------------------------------
  constexpr auto n_streams = 1;

  std::array<cudaStream_t, n_streams> streams;
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  }

  auto pipe = std::make_unique<NaivePipe>(params.n);

  gpu::dispatch_InitRandomVec4(params.n_blocks,
                               streams[0],
                               pipe->u_points.data(),
                               params.n,
                               params.min_coord,
                               params.getRange(),
                               params.seed);

  gpu::dispatch_ComputeMorton(params.n_blocks,
                              streams[0],
                              pipe->u_points.data(),
                              pipe->u_morton_keys.data(),
                              params.n,
                              params.min_coord,
                              params.getRange());

  SYNC_STREAM(streams[0]);

  // ------------------------------

  // peek 1024 morton keys

  for (auto i = 0; i < 32; ++i) {
    std::cout << i << ":\t" << u_morton_keys[i] << std::endl;
  }

  // const std::array<std::unique_ptr<Pipe>, n_streams> pipes{
  //     std::make_unique<Pipe>(params.n),
  // };

  // for (auto i = 0; i < n_streams; ++i) {
  //   pipes[i]->attachStream(streams[i]);
  // }

  // {
  //   gpu::dispatchKernel(gpu::k_InitRandomVec4,
  //                       params.n_blocks,
  //                       streams[0],
  //                       // *** original arguments start ***
  //                       pipes[0]->getPoints(),
  //                       params.n,
  //                       params.min_coord,
  //                       params.getRange(),
  //                       params.seed);

  //   gpu::dispatchKernel(gpu::k_ComputeMortonCode,
  //                       params.n_blocks,
  //                       streams[0],
  //                       // *** original arguments start ***
  //                       pipes[0]->getPoints(),
  //                       pipes[0]->getMortonKeys(),
  //                       params.n,
  //                       params.min_coord,
  //                       params.getRange());

  //   gpu::dispatch_RadixSort(params.n_blocks, streams[0], pipes[0]->onesweep);

  //   SYNC_STREAM(streams[0]);
  // }

  // // ------------------------------

  // auto is_sorted = std::is_sorted(pipes[0]->onesweep.getSort(),
  //                                 pipes[0]->onesweep.getSort() + params.n);
  // spdlog::info("is_sorted: {}", is_sorted);

  spdlog::info("Done");
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  }
  return 0;
}
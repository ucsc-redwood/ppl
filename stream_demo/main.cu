#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <glm/glm.hpp>

#include "cuda/helper.cuh"
#include "cuda/kernels/01_morton.cuh"
#include "device_dispatcher.h"
#include "handlers/pipe.h"
#include "host_dispatcher.h"
#include "openmp/kernels/00_init.hpp"

int DivideAndRoundUp(const int a, const int b) { return (a + b - 1) / b; }

[[maybe_unused]] void attachPipeToStream(const cudaStream_t stream,
                                         const Pipe* p) {
  // Pipe
  ATTACH_STREAM_SINGLE(p->u_points);
  ATTACH_STREAM_SINGLE(p->u_edge_count);
  ATTACH_STREAM_SINGLE(p->u_edge_offset);

  // One sweep
  ATTACH_STREAM_SINGLE(p->sort.u_sort);
  ATTACH_STREAM_SINGLE(p->sort.u_sort_alt);

  // Unique
  ATTACH_STREAM_SINGLE(p->unique.u_keys_out);

  // Radix tree
  ATTACH_STREAM_SINGLE(p->brt.u_prefix_n);
  ATTACH_STREAM_SINGLE(p->brt.u_has_leaf_left);
  ATTACH_STREAM_SINGLE(p->brt.u_has_leaf_right);
  ATTACH_STREAM_SINGLE(p->brt.u_left_child);
  ATTACH_STREAM_SINGLE(p->brt.u_parent);

  // Octree
  ATTACH_STREAM_SINGLE(p->oct.u_children);
  ATTACH_STREAM_SINGLE(p->oct.u_corner);
  ATTACH_STREAM_SINGLE(p->oct.u_cell_size);
  ATTACH_STREAM_SINGLE(p->oct.u_child_node_mask);
  ATTACH_STREAM_SINGLE(p->oct.u_child_leaf_mask);

  SYNC_STREAM(stream);
}

// __global__ void k_DoSomethingA(glm::vec4* u_input,
//                                unsigned int* u_output,
//                                const int n) {
//   const auto i = threadIdx.x + blockIdx.x * blockDim.x;
//   const auto stride = blockDim.x * gridDim.x;

//   for (auto j = i; j < n; j += stride) {
//     u_output[j] = static_cast<unsigned int>(sqrt(u_input[j].x));
//   }
// }

// __global__ void k_DoSomethingB(glm::vec4* u_input,
//                                unsigned int* u_output,
//                                const int n) {
//   const auto i = threadIdx.x + blockIdx.x * blockDim.x;
//   const auto stride = blockDim.x * gridDim.x;

//   for (auto j = i; j < n; j += stride) {
//     u_output[j] = u_input[j].x * u_input[j].y * u_input[j].z * u_input[j].w;
//   }
// }

// __global__ void k_DoSomethingC(glm::vec4* u_input,
//                                unsigned int* u_output,
//                                const int n) {
//   const auto i = threadIdx.x + blockIdx.x * blockDim.x;
//   const auto stride = blockDim.x * gridDim.x;

//   for (auto j = i; j < n; j += stride) {
//     u_output[j] = static_cast<unsigned int>(exp(u_input[j].z));
//   }
// }

// struct Task {
//   void allocate(const int n) {
//     this->n = n;
//     CHECK_CUDA_CALL(cudaMallocManaged(&u_input, n * sizeof(glm::vec4)));
//     CHECK_CUDA_CALL(cudaMallocManaged(&u_output, n * sizeof(unsigned int)));

//     std::generate(u_input, u_input + n, []() {
//       return glm::vec4{1.0f, 2.0f, 3.0f, 4.0f};
//     });
//   }

//   ~Task() {
//     CUDA_FREE(u_input);
//     CUDA_FREE(u_output);
//   }

//   int n;
//   glm::vec4* u_input;
//   unsigned int* u_output;
// };

// void execute(Task& t, cudaStream_t* stream, const int tid) {
//   CHECK_CUDA_CALL(
//       cudaStreamAttachMemAsync(stream[tid], t.u_input, 0,
//       cudaMemAttachSingle));

//   if (tid == 0) {
//     k_DoSomethingA<<<1, 512, 0, stream[tid]>>>(t.u_input, t.u_output, t.n);
//   } else if (tid == 1) {
//     k_DoSomethingB<<<1, 512, 0, stream[tid]>>>(t.u_input, t.u_output, t.n);
//   } else if (tid == 2) {
//     k_DoSomethingC<<<1, 512, 0, stream[tid]>>>(t.u_input, t.u_output, t.n);
//   } else {
//     gpu::k_ComputeMortonCode<<<1, 512, 0, stream[tid]>>>(
//         t.u_input, t.u_output, t.n, 0.0f, 100.0f);
//   }
// }

glm::vec4* u_original_input;

struct Task {
  Pipe* p;
};

void execute(Task& p, cudaStream_t* stream, const int stream_id) {
  CHECK_CUDA_CALL(cudaMemcpyAsync(p.p->u_points,
                                  u_original_input,
                                  p.p->getInputSize() * sizeof(glm::vec4),
                                  cudaMemcpyDefault,
                                  stream[stream_id]));
  // p.p->acquireNextFrameData();

  gpu::v2::dispatch_ComputeMorton(1, stream[stream_id], *p.p);

  // gpu::v2::dispatch_RadixSort(params.n_blocks, stream, *pipe);
  // gpu::v2::dispatch_RemoveDuplicates(params.n_blocks, stream, *pipe);
  // gpu::v2::dispatch_BuildRadixTree(params.n_blocks, stream, *pipe);
  // gpu::v2::dispatch_EdgeCount(params.n_blocks, stream, *pipe);
  // gpu::v2::dispatch_EdgeOffset(params.n_blocks, stream, *pipe);
  // gpu::v2::dispatch_BuildOctree(params.n_blocks, stream, *pipe);
}

int main() {
  constexpr auto n = 1 << 20;  // 1M elements

  constexpr auto n_tasks = 120;

  CHECK_CUDA_CALL(cudaMallocManaged(&u_original_input, n * sizeof(glm::vec4)));
  cpu::k_InitRandomVec4(u_original_input, n, 0.0f, 1024.0f, 114514);

  // std::array<Pipe, n_streams> pipes{Pipe(n, 0.0f, 1024.0f, 1),
  //                                   Pipe(n, 0.0f, 1024.0f, 2),
  //                                   Pipe(n, 0.0f, 1024.0f, 3),
  //                                   Pipe(n, 0.0f, 1024.0f, 4)};

  constexpr auto n_streams = 4;
  std::array<cudaStream_t, n_streams> streams;

  // std::vector<Task> tasks(n_tasks);  // to dos
  // for (auto& task : tasks) {
  //   task.allocate(n);
  // }

  spdlog::set_level(spdlog::level::trace);

  std::vector<Task> pipes(n_streams);
  for (auto i = 0; i < n_streams; ++i) {
    pipes[i].p = new Pipe(n, 0.0f, 1024.0f, 114514);
  }

  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  }

  constexpr auto min_coord = 0.0f;
  constexpr auto range = 100.0f;

  cudaEvent_t start, stop;
  CHECK_CUDA_CALL(cudaEventCreate(&start));
  CHECK_CUDA_CALL(cudaEventCreate(&stop));

  CHECK_CUDA_CALL(cudaEventRecord(start, nullptr));

  // ------------------------------

  const auto n_iters = DivideAndRoundUp(n_tasks, n_streams);

  // // baseline. 1 stream handles all tasks
  // for (auto i = 0; i < n_tasks; ++i) {
  //   execute(pipes[0], streams.data(), 0);
  // }

  for (auto i = 0; i < n_iters; ++i) {
    for (auto stream_id = 0; stream_id < n_streams; ++stream_id) {
      execute(pipes[stream_id], streams.data(), stream_id);
    }
    SYNC_DEVICE();
  }

  // for (auto i = 0; i < n_tasks; ++i) {
  //   execute(pipes[0], streams.data(), 0);
  //   SYNC_DEVICE();
  // }

  // for (auto i = 0; i < n_iters; ++i) {
  //   // SYNC_STREAM(streams[0]);
  //   for (auto stream_id = 0; stream_id < n_streams; ++stream_id) {
  //     execute(pipes[i], streams.data(), stream_id);
  //     SYNC_DEVICE();
  //   }
  // }

  // for (auto i = 0; i < n_tasks; ++i) {
  //   const auto my_id = i % n_streams;
  //   CHECK_CUDA_CALL(cudaMemcpyAsync(tasks[my_id].u_input,
  //                                   u_original_input,
  //                                   n * sizeof(glm::vec4),
  //                                   cudaMemcpyDefault,
  //                                   streams[my_id]));

  //   execute(tasks[my_id], streams.data(), my_id);
  // }

  // ------------------------------

  SYNC_DEVICE();

  CHECK_CUDA_CALL(cudaEventRecord(stop, nullptr));
  CHECK_CUDA_CALL(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  spdlog::info("Total time: {} ms", milliseconds);

  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  }

  CHECK_CUDA_CALL(cudaEventDestroy(start));
  CHECK_CUDA_CALL(cudaEventDestroy(stop));

  // delete[] h_original_input;

  CUDA_FREE(u_original_input);

  return 0;
}
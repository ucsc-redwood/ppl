#include <driver_types.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <deque>
#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "cuda/agents/prefix_sum_agent.cuh"
#include "cuda/agents/unique_agent.cuh"
#include "cuda/helper.cuh"
#include "cuda/kernels/01_morton.cuh"
#include "cuda/kernels/02_sort.cuh"
#include "cuda/kernels/03_unique.cuh"
#include "cuda/kernels/06_prefix_sum.cuh"
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

glm::vec4* u_original_input;

struct Task {
  Pipe* p;
};

enum class Method {
  OneStreamHandleAll = 0,
  FourStreamsHandleAll = 1,
  SimpleGpuPipeline = 2,
  CpuGpuHalfHalf = 3,
  SimpleCpuGpuPipeline = 4,
};

const std::array<std::string, 5> methodNames{
    "OneStreamHandleAll",
    "FourStreamsHandleAll",
    "SimpleGpuPipeline",
    "CpuGpuHalfHalf",
    "SimpleCpuGpuPipeline",
};

void getInputFrame(Pipe* p, cudaStream_t s) {
  CHECK_CUDA_CALL(cudaMemcpyAsync(p->u_points,
                                  u_original_input,
                                  p->getInputSize() * sizeof(glm::vec4),
                                  cudaMemcpyDefault,
                                  s));
}

void gpu_dispatch_RemoveDuplicatesNoSync(int grid_size,
                                         cudaStream_t stream,
                                         Pipe& pipe) {
  constexpr auto unique_block_size = gpu::UniqueAgent::n_threads;  // 256
  constexpr auto prefix_block_size =
      gpu::PrefixSumAgent<unsigned int>::n_threads;  // 128

  gpu::k_FindDups<<<grid_size, unique_block_size, 0, stream>>>(
      pipe.getSortedKeys(),
      pipe.unique.im_storage.u_flag_heads,
      pipe.getInputSize());

  gpu::k_SingleBlockExclusiveScan<<<1, prefix_block_size, 0, stream>>>(
      pipe.unique.im_storage.u_flag_heads,
      pipe.unique.im_storage.u_flag_heads,
      pipe.getInputSize());

  gpu::k_MoveDups<<<grid_size, unique_block_size, 0, stream>>>(
      pipe.getSortedKeys(),
      pipe.unique.im_storage.u_flag_heads,
      pipe.getInputSize(),
      pipe.unique.u_keys_out,
      nullptr);

  pipe.n_unique_keys = 0.9 * pipe.getInputSize();
  pipe.n_brt_nodes = pipe.n_unique_keys - 1;
}

// Baselines
// 1. One stream, handle all
void run_one_stream_handle_all(const int n, const int n_tasks) {
  auto p = new Pipe(n, 0.0f, 1024.0f, 114514);
  getInputFrame(p, nullptr);

  cudaStream_t s;
  CHECK_CUDA_CALL(cudaStreamCreate(&s));

  cudaEvent_t start, stop;
  CHECK_CUDA_CALL(cudaEventCreate(&start));
  CHECK_CUDA_CALL(cudaEventCreate(&stop));
  CHECK_CUDA_CALL(cudaEventRecord(start, nullptr));

  for (auto i = 0; i < n_tasks; ++i) {
    // p->acquireNextFrameData();
    gpu::v2::dispatch_ComputeMorton(6, s, *p);
    gpu::v2::dispatch_RadixSort(6, s, *p);
    gpu_dispatch_RemoveDuplicatesNoSync(6, s, *p);  //
    gpu::v2::dispatch_BuildRadixTree(6, s, *p);
  }

  CHECK_CUDA_CALL(cudaEventRecord(stop, nullptr));
  CHECK_CUDA_CALL(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  spdlog::info("Total time: {} ms", milliseconds);

  CHECK_CUDA_CALL(cudaStreamDestroy(s));
}

// Baselines
// 1. One stream, handle all
void run_four_stream_handle_all(const int n, const int n_tasks) {
  // std::vector<Task> tasks(n_tasks);
  // for (auto i = 0; i < n_tasks; ++i) {
  //   tasks[i].p = new Pipe(n, 0.0f, 1024.0f, 114514);
  //   getInputFrame(tasks[i].p, nullptr);
  // }

  // constexpr auto n_streams = 4;
  // std::array<cudaStream_t, n_streams> streams;
  // for (auto& stream : streams) {
  //   CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  // }

  // cudaEvent_t start, stop;
  // CHECK_CUDA_CALL(cudaEventCreate(&start));
  // CHECK_CUDA_CALL(cudaEventCreate(&stop));
  // CHECK_CUDA_CALL(cudaEventRecord(start, nullptr));

  // for (auto i = 0; i < n_tasks; ++i) {
  //   const auto my_stream_id = i % n_streams;

  //   auto p = tasks[i].p;
  //   // p->acquireNextFrameData();
  //   gpu::v2::dispatch_ComputeMorton(6, streams[my_stream_id], *p);
  //   gpu::v2::dispatch_RadixSort(6, streams[my_stream_id], *p);
  //   gpu::v2::dispatch_RemoveDuplicates(6, streams[my_stream_id], *p);
  //   gpu::v2::dispatch_BuildRadixTree(6, streams[my_stream_id], *p);
  // }

  // SYNC_DEVICE();

  // CHECK_CUDA_CALL(cudaEventRecord(stop, nullptr));
  // CHECK_CUDA_CALL(cudaEventSynchronize(stop));

  // float milliseconds = 0;
  // CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  // spdlog::info("Total time: {} ms", milliseconds);
  // CHECK_CUDA_CALL(cudaEventDestroy(start));
  // CHECK_CUDA_CALL(cudaEventDestroy(stop));
  // for (auto& stream : streams) {
  //   CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  // }
}

void run_redwood_cpu_gpu(const int n, const int n_tasks) {
  cudaStream_t s;
  CHECK_CUDA_CALL(cudaStreamCreate(&s));

  Task cpu_task;
  cpu_task.p = new Pipe(n, 0.0f, 1024.0f, 114514);
  getInputFrame(cpu_task.p, nullptr);

  Task gpu_task;
  gpu_task.p = new Pipe(n, 0.0f, 1024.0f, 114514);
  getInputFrame(gpu_task.p, nullptr);

  cudaEvent_t start, stop;
  CHECK_CUDA_CALL(cudaEventCreate(&start));
  CHECK_CUDA_CALL(cudaEventCreate(&stop));

  CHECK_CUDA_CALL(cudaEventRecord(start, s));

  for (auto i = 0; i < n_tasks / 2; ++i) {
    gpu::v2::dispatch_ComputeMorton(6, s, *gpu_task.p);
    gpu::v2::dispatch_RadixSort(6, s, *gpu_task.p);
    gpu_dispatch_RemoveDuplicatesNoSync(6, s, *gpu_task.p);
    gpu::v2::dispatch_BuildRadixTree(6, s, *gpu_task.p);

    cpu::v2::dispatch_ComputeMorton(6, *cpu_task.p);
    cpu::v2::dispatch_RadixSort(6, *cpu_task.p);
    cpu::v2::dispatch_RemoveDuplicates(6, *cpu_task.p);
    cpu::v2::dispatch_BuildRadixTree(6, *cpu_task.p);
  }

  CHECK_CUDA_CALL(cudaEventRecord(stop, s));
  CHECK_CUDA_CALL(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  spdlog::info("Total time: {} ms", milliseconds);

  delete cpu_task.p;
  delete gpu_task.p;

  CHECK_CUDA_CALL(cudaStreamDestroy(s));
}

void run_simple_gpu_pipeline(const int n, const int n_tasks) {
  constexpr auto n_streams = 4;

  std::vector<Task> gpu_task(n_streams);
  for (auto i = 0; i < n_streams; ++i) {
    gpu_task[i].p = new Pipe(n, 0.0f, 1024.0f, 114514);
    getInputFrame(gpu_task[i].p, nullptr);
  }

  // setup initial
  for (auto i = 0; i < n_streams; ++i) {
    auto p = gpu_task[i].p;
    cpu::v2::dispatch_ComputeMorton(1, *p);
    cpu::v2::dispatch_RadixSort(2, *p);
    cpu::v2::dispatch_RemoveDuplicates(1, *p);
    cpu::v2::dispatch_BuildRadixTree(2, *p);
  }

  spdlog::info("=============First few stages done====================");

  std::array<cudaStream_t, n_streams> streams;
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  }

  cudaEvent_t start, stop;
  CHECK_CUDA_CALL(cudaEventCreate(&start));
  CHECK_CUDA_CALL(cudaEventCreate(&stop));
  CHECK_CUDA_CALL(cudaEventRecord(start, nullptr));

  for (int i = 3; i < n_tasks; ++i) {
    auto first_pipe = gpu_task[i % n_streams].p;
    auto sec_pipe = gpu_task[(i - 1) % n_streams].p;
    auto third_pipe = gpu_task[(i - 2) % n_streams].p;
    auto fourth_pipe = gpu_task[(i - 3) % n_streams].p;

    gpu::v2::dispatch_ComputeMorton(1, streams[0], *first_pipe);
    gpu::v2::dispatch_RadixSort(2, streams[1], *sec_pipe);
    gpu_dispatch_RemoveDuplicatesNoSync(1, streams[2], *third_pipe);
    gpu::v2::dispatch_BuildRadixTree(2, streams[3], *fourth_pipe);
  }

  SYNC_DEVICE();

  CHECK_CUDA_CALL(cudaEventRecord(stop, nullptr));
  CHECK_CUDA_CALL(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  spdlog::info("Total time: {} ms", milliseconds);

  CHECK_CUDA_CALL(cudaEventDestroy(start));
  CHECK_CUDA_CALL(cudaEventDestroy(stop));
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  }
}

void run_simple_cpu_gpu_pipeline(const int n, const int n_tasks) {
  std::vector<Task> tasks(n_tasks);
  for (auto i = 0; i < n_tasks; ++i) {
    tasks[i].p = new Pipe(n, 0.0f, 1024.0f, 114514);
    getInputFrame(tasks[i].p, nullptr);
  }

  constexpr auto n_streams = 4;
  std::array<cudaStream_t, n_streams> streams;
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamCreate(&stream));
  }

  cudaEvent_t start, stop;
  CHECK_CUDA_CALL(cudaEventCreate(&start));
  CHECK_CUDA_CALL(cudaEventCreate(&stop));
  CHECK_CUDA_CALL(cudaEventRecord(start, nullptr));

  // start with 1
  // for (auto i = 1; i < n_tasks; ++i) {
  // each stream process a different kernel

  // ------------------------------
  // need to setup the first few (stages)
  //  ------------------------------

  {
    auto first_pipe = tasks[0].p;
    cpu::v2::dispatch_ComputeMorton(1, *first_pipe);
    cpu::v2::dispatch_RadixSort(2, *first_pipe);
    cpu::v2::dispatch_RemoveDuplicates(1, *first_pipe);
    cpu::v2::dispatch_BuildRadixTree(2, *first_pipe);

    auto sec_pipe = tasks[1].p;
    cpu::v2::dispatch_ComputeMorton(1, *sec_pipe);
    cpu::v2::dispatch_RadixSort(2, *sec_pipe);
    cpu::v2::dispatch_RemoveDuplicates(1, *sec_pipe);
    cpu::v2::dispatch_BuildRadixTree(2, *sec_pipe);

    auto third_pipe = tasks[2].p;
    cpu::v2::dispatch_ComputeMorton(1, *third_pipe);
    cpu::v2::dispatch_RadixSort(2, *third_pipe);
    cpu::v2::dispatch_RemoveDuplicates(1, *third_pipe);
    cpu::v2::dispatch_BuildRadixTree(2, *third_pipe);

    auto fourth_pipe = tasks[3].p;
    cpu::v2::dispatch_ComputeMorton(1, *fourth_pipe);
    cpu::v2::dispatch_RadixSort(2, *fourth_pipe);
    cpu::v2::dispatch_RemoveDuplicates(1, *fourth_pipe);
    cpu::v2::dispatch_BuildRadixTree(2, *fourth_pipe);

    SYNC_DEVICE();
  }

  spdlog::info("=============First few stages done====================");

  // ------------------------------
  // end of setup
  // ------------------------------

  for (auto i = 3; i < n_tasks; ++i) {
    auto first_pipe = tasks[i].p;
    auto sec_pipe = tasks[i - 1].p;
    auto third_pipe = tasks[i - 2].p;
    auto fourth_pipe = tasks[i - 3].p;

    gpu::v2::dispatch_ComputeMorton(1, streams[0], *first_pipe);
    gpu::v2::dispatch_RadixSort(2, streams[1], *sec_pipe);
    cpu::v2::dispatch_BuildRadixTree(6, *fourth_pipe);
    gpu::v2::dispatch_RemoveDuplicates(1, streams[2], *third_pipe);
  }

  SYNC_DEVICE();

  CHECK_CUDA_CALL(cudaEventRecord(stop, nullptr));
  CHECK_CUDA_CALL(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  spdlog::info("Total time: {} ms", milliseconds);

  CHECK_CUDA_CALL(cudaEventDestroy(start));
  CHECK_CUDA_CALL(cudaEventDestroy(stop));
  for (auto& stream : streams) {
    CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  }
}

int main(const int argc, const char* argv[]) {
  constexpr auto n = 1920 * 1080;  // ~300k elements
  constexpr auto n_tasks = 40;

  Method method = Method::OneStreamHandleAll;
  if (argc > 1) {
    method = static_cast<Method>(std::stoi(argv[1]));
  }

  spdlog::info("Method: {}", methodNames[static_cast<int>(method)]);

  // spdlog::set_level(spdlog::level::debug);
  spdlog::set_level(spdlog::level::info);

  CHECK_CUDA_CALL(cudaMallocManaged(&u_original_input, n * sizeof(glm::vec4)));
  cpu::k_InitRandomVec4(u_original_input, n, 0.0f, 1024.0f, 114514);

  switch (method) {
    case Method::OneStreamHandleAll:  // 0
      run_one_stream_handle_all(n, n_tasks);
      break;
    case Method::FourStreamsHandleAll:  // 1
      run_four_stream_handle_all(n, n_tasks);
      break;
    case Method::SimpleGpuPipeline:  // 2
      run_simple_gpu_pipeline(n, n_tasks);
      break;
    case Method::CpuGpuHalfHalf:  // 3
      run_redwood_cpu_gpu(n, n_tasks);
      break;
    case Method::SimpleCpuGpuPipeline:  // 4
      run_simple_cpu_gpu_pipeline(n, n_tasks);
      break;
    default:
      throw std::runtime_error("Unknown method");
  }

  // ------------------------------

  // // ------------------------------

  // const auto n_iters = DivideAndRoundUp(n_tasks, n_streams);

  // for (auto i = 0; i < n_tasks; ++i) {
  //   const auto my_stream_id = i % n_streams;
  //   execute(tasks[i], streams.data(), my_stream_id);
  // }

  // // ------------------------------

  // SYNC_DEVICE();

  CUDA_FREE(u_original_input);
  return 0;
}
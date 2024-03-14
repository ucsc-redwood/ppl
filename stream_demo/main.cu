#include <driver_types.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <deque>
#include <glm/glm.hpp>
#include <string>

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

glm::vec4* u_original_input;

struct Task {
  Pipe* p;
};

enum class Method {
  OneStreamHandleAll = 0,
  FourStreamsHandleAll = 1,
  SimpleGpuPipeline = 2,
  CpuGpuHalfHalf = 3,
};

const std::array<std::string, 3> methodNames{
    "OneStreamHandleAll",
    "FourStreamsHandleAll",
    "SimpleGpuPipeline",
};

void getInputFrame(Pipe* p, cudaStream_t s) {
  CHECK_CUDA_CALL(cudaMemcpyAsync(p->u_points,
                                  u_original_input,
                                  p->getInputSize() * sizeof(glm::vec4),
                                  cudaMemcpyDefault,
                                  s));
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
    p->acquireNextFrameData();
    gpu::v2::dispatch_ComputeMorton(1, s, *p);
    gpu::v2::dispatch_RadixSort(2, s, *p);
    gpu::v2::dispatch_RemoveDuplicates(1, s, *p);
    gpu::v2::dispatch_BuildRadixTree(2, s, *p);
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

  for (auto i = 0; i < n_tasks; ++i) {
    const auto my_stream_id = i % n_streams;

    auto p = tasks[i].p;
    p->acquireNextFrameData();
    gpu::v2::dispatch_ComputeMorton(1, streams[my_stream_id], *p);
    gpu::v2::dispatch_RadixSort(2, streams[my_stream_id], *p);
    gpu::v2::dispatch_RemoveDuplicates(1, streams[my_stream_id], *p);
    gpu::v2::dispatch_BuildRadixTree(2, streams[my_stream_id], *p);
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

void run_cpu_gpu_half_half(const int n, const int n_tasks) {
  std::vector<Task> cpu_tasks(n_tasks / 2);
  std::vector<Task> gpu_tasks(n_tasks / 2);

  for (auto i = 0; i < n_tasks / 2; ++i) {
    cpu_tasks[i].p = new Pipe(n, 0.0f, 1024.0f, 114514);
    getInputFrame(cpu_tasks[i].p, nullptr);
  }

  for (auto i = 0; i < n_tasks / 2; ++i) {
    gpu_tasks[i].p = new Pipe(n, 0.0f, 1024.0f, 114514);
    getInputFrame(gpu_tasks[i].p, nullptr);
  }

  cudaStream_t s;
  CHECK_CUDA_CALL(cudaStreamCreate(&s));

  cudaEvent_t start, stop;
  CHECK_CUDA_CALL(cudaEventCreate(&start));
  CHECK_CUDA_CALL(cudaEventCreate(&stop));

  CHECK_CUDA_CALL(cudaEventRecord(start, s));

  for (auto i = 0; i < n_tasks / 2; ++i) {
    cpu_tasks[i].p->acquireNextFrameData();
    gpu_tasks[i].p->acquireNextFrameData();

    gpu::v2::dispatch_ComputeMorton(1, s, *cpu_tasks[i].p);
    gpu::v2::dispatch_RadixSort(2, s, *cpu_tasks[i].p);
    gpu::v2::dispatch_RemoveDuplicates(1, s, *cpu_tasks[i].p);
    gpu::v2::dispatch_BuildRadixTree(2, s, *cpu_tasks[i].p);

    cpu::v2::dispatch_ComputeMorton(1, *gpu_tasks[i].p);
    cpu::v2::dispatch_RadixSort(2, *gpu_tasks[i].p);
    cpu::v2::dispatch_RemoveDuplicates(1, *gpu_tasks[i].p);
    cpu::v2::dispatch_BuildRadixTree(2, *gpu_tasks[i].p);
  }

  CHECK_CUDA_CALL(cudaStreamDestroy(s));
}

void run_simple_gpu_pipeline(const int n, const int n_tasks) {
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
    // auto first_pipe = tasks[0].p;
    // gpu::v2::dispatch_ComputeMorton(1, streams[0], *first_pipe);
    // gpu::v2::dispatch_RadixSort(2, streams[0], *first_pipe);
    // gpu::v2::dispatch_RemoveDuplicates(1, streams[0], *first_pipe);
    // gpu::v2::dispatch_BuildRadixTree(2, streams[0], *first_pipe);

    // auto sec_pipe = tasks[1].p;
    // gpu::v2::dispatch_ComputeMorton(1, streams[0], *sec_pipe);
    // gpu::v2::dispatch_RadixSort(2, streams[0], *sec_pipe);
    // gpu::v2::dispatch_RemoveDuplicates(1, streams[0], *sec_pipe);
    // gpu::v2::dispatch_BuildRadixTree(2, streams[0], *sec_pipe);

    // auto third_pipe = tasks[2].p;
    // gpu::v2::dispatch_ComputeMorton(1, streams[0], *third_pipe);
    // gpu::v2::dispatch_RadixSort(2, streams[0], *third_pipe);
    // gpu::v2::dispatch_RemoveDuplicates(1, streams[0], *third_pipe);
    // gpu::v2::dispatch_BuildRadixTree(2, streams[0], *third_pipe);

    // auto fourth_pipe = tasks[3].p;
    // gpu::v2::dispatch_ComputeMorton(1, streams[0], *fourth_pipe);
    // gpu::v2::dispatch_RadixSort(2, streams[0], *fourth_pipe);
    // gpu::v2::dispatch_RemoveDuplicates(1, streams[0], *fourth_pipe);
    // gpu::v2::dispatch_BuildRadixTree(2, streams[0], *fourth_pipe);

    auto first_pipe = tasks[0].p;
    cpu::v2::dispatch_ComputeMorton(1, *first_pipe);
    cpu::v2::dispatch_RadixSort(2, *first_pipe);
    cpu::v2::dispatch_RemoveDuplicates(1, *first_pipe);
    cpu::v2::dispatch_BuildRadixTree(2, *first_pipe);

    // peek 5 sorted morton
    for (auto i = 0; i < 5; ++i) {
      spdlog::trace("{}", first_pipe->unique.u_keys_out[i]);
    }

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

  // stream 0 for morton
  // stream 1 for radix sort
  // const auto i = 3;

  for (auto i = 3; i < n_tasks; ++i) {
    auto first_pipe = tasks[i].p;
    auto sec_pipe = tasks[i - 1].p;
    auto third_pipe = tasks[i - 2].p;
    auto fourth_pipe = tasks[i - 3].p;

    gpu::v2::dispatch_ComputeMorton(1, streams[0], *first_pipe);
    gpu::v2::dispatch_RadixSort(2, streams[1], *sec_pipe);
    gpu::v2::dispatch_RemoveDuplicates(1, streams[2], *third_pipe);
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

int main(const int argc, const char* argv[]) {
  constexpr auto n = 640 * 480;  // ~300k elements
  constexpr auto n_tasks = 80;

  Method method = Method::OneStreamHandleAll;
  if (argc > 1) {
    method = static_cast<Method>(std::stoi(argv[1]));
  }

  spdlog::info("Method: {}", methodNames[static_cast<int>(method)]);

  spdlog::set_level(spdlog::level::debug);

  CHECK_CUDA_CALL(cudaMallocManaged(&u_original_input, n * sizeof(glm::vec4)));
  cpu::k_InitRandomVec4(u_original_input, n, 0.0f, 1024.0f, 114514);

  switch (method) {
    case Method::OneStreamHandleAll:
      run_one_stream_handle_all(n, n_tasks);
      break;
    case Method::FourStreamsHandleAll:
      run_four_stream_handle_all(n, n_tasks);
      break;
    case Method::SimpleGpuPipeline:
      run_simple_gpu_pipeline(n, n_tasks);
      break;
    case Method::CpuGpuHalfHalf:
      run_cpu_gpu_half_half(n, n_tasks);
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
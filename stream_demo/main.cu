#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <glm/glm.hpp>

#include "cuda/helper.cuh"
#include "cuda/kernels/01_morton.cuh"

__global__ void k_DoSomethingA(glm::vec4* u_input,
                               unsigned int* u_output,
                               const int n) {
  const auto i = threadIdx.x + blockIdx.x * blockDim.x;
  const auto stride = blockDim.x * gridDim.x;

  for (auto j = i; j < n; j += stride) {
    u_output[j] = static_cast<unsigned int>(sqrt(u_input[j].x));
  }
}

__global__ void k_DoSomethingB(glm::vec4* u_input,
                               unsigned int* u_output,
                               const int n) {
  const auto i = threadIdx.x + blockIdx.x * blockDim.x;
  const auto stride = blockDim.x * gridDim.x;

  for (auto j = i; j < n; j += stride) {
    u_output[j] = static_cast<unsigned int>(pow(u_input[j].y, 6.66));
  }
}

__global__ void k_DoSomethingC(glm::vec4* u_input,
                               unsigned int* u_output,
                               const int n) {
  const auto i = threadIdx.x + blockIdx.x * blockDim.x;
  const auto stride = blockDim.x * gridDim.x;

  for (auto j = i; j < n; j += stride) {
    u_output[j] = static_cast<unsigned int>(exp(u_input[j].z));
  }
}

struct Task {
  void allocate(const int n) {
    this->n = n;
    MALLOC_MANAGED(&u_input, n);
    MALLOC_MANAGED(&u_output, n);

    std::generate(u_input, u_input + n, []() {
      return glm::vec4{1.0f, 2.0f, 3.0f, 4.0f};
    });
  }

  ~Task() {
    CUDA_FREE(u_input);
    CUDA_FREE(u_output);
  }

  int n;
  glm::vec4* u_input;
  unsigned int* u_output;
};

void execute(Task& t, cudaStream_t* stream, const int tid) {
  CHECK_CUDA_CALL(
      cudaStreamAttachMemAsync(stream[tid], t.u_input, 0, cudaMemAttachSingle));

  if (tid == 0) {
    k_DoSomethingA<<<1, 512, 0, stream[tid]>>>(t.u_input, t.u_output, t.n);
  } else if (tid == 1) {
    k_DoSomethingB<<<1, 512, 0, stream[tid]>>>(t.u_input, t.u_output, t.n);
  } else if (tid == 2) {
    k_DoSomethingC<<<1, 512, 0, stream[tid]>>>(t.u_input, t.u_output, t.n);
  } else {
    gpu::k_ComputeMortonCode<<<1, 512, 0, stream[tid]>>>(
        t.u_input, t.u_output, t.n, 0.0f, 100.0f);
  }
}

int main() {
  constexpr auto n = 1 << 20;  // 1M elements

  constexpr auto n_tasks = 100;

  volatile auto h_original_input = new glm::vec4[n];
  std::generate(h_original_input, h_original_input + n, []() {
    return glm::vec4{1.0f, 2.0f, 3.0f, 4.0f};
  });

  constexpr auto n_streams = 4;
  std::array<cudaStream_t, n_streams> streams;

  //   std::array<Task, n_streams> tasks{Task(n), Task(n), Task(n), Task(n)};

  //   std::vector<Task> tasks(n_tasks);  // to dos
  //   for (auto& task : tasks) {
  //     task.allocate(n);
  //   }

  std::array<Task, n_streams> tasks{Task(), Task(), Task(), Task()};

  for (auto& task : tasks) {
    task.allocate(n);
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

  for (auto i = 0; i < n_tasks; ++i) {
    const auto my_id = i % n_streams;
    std::copy_n(h_original_input, n, tasks[my_id].u_input);

    execute(tasks[my_id], streams.data(), my_id);
  }

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

  delete[] h_original_input;
  return 0;
}
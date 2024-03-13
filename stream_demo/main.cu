#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <glm/glm.hpp>

#include "cuda/helper.cuh"
#include "cuda/kernels/01_morton.cuh"

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

  gpu::k_ComputeMortonCode<<<1, 768, 0, stream[tid]>>>(
      t.u_input, t.u_output, t.n, 0.0f, 100.0f);
}

int main() {
  constexpr auto n = 1 << 20;  // 1M elements

  constexpr auto n_tasks = 100;

  constexpr auto n_streams = 4;
  std::array<cudaStream_t, n_streams> streams;

  //   std::array<Task, n_streams> tasks{Task(n), Task(n), Task(n), Task(n)};

  std::vector<Task> tasks(n_tasks);  // to dos
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
    execute(tasks[i], streams.data(), i % n_streams);
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

  return 0;
}
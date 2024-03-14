#include <pthread.h>

#include <glm/glm.hpp>

#include "shared/morton.h"

namespace cpu {

typedef struct {
  const glm::vec4* data;
  unsigned int* morton_keys;
  int start;
  int end;
  float min_coord;
  float range;
} MortonCodeArgs;

void* computeMortonCode(void* args) {
  MortonCodeArgs* tData = (MortonCodeArgs*)args;
  for (int i = tData->start; i < tData->end; i++) {
    tData->morton_keys[i] =
        shared::xyz_to_morton32(tData->data[i], tData->min_coord, tData->range);
  }
}

void k_ComputeMortonCode(const int thread_start,
                         const int thread_end,
                         const glm::vec4* data,
                         unsigned int* morton_keys,
                         const int n,
                         const float min_coord,
                         const float range) {
  const int n_threads = thread_end - thread_start;

  pthread_t threads[n_threads];
  MortonCodeArgs threadData[n_threads];
  int chunkSize = n / n_threads;

  for (int i = 0; i < n_threads; i++) {
    threadData[i].data = data;
    threadData[i].morton_keys = morton_keys;
    threadData[i].start = i * chunkSize;
    threadData[i].end =
        (i == n_threads - 1)
            ? n
            : (i + 1) * chunkSize;  // Last thread may have more work
    threadData[i].min_coord = min_coord;
    threadData[i].range = range;

    pthread_create(&threads[i], NULL, computeMortonCode, (void*)&threadData[i]);
  }
  // Join threads
  for (int i = 0; i < n_threads; i++) {
    pthread_join(threads[i], NULL);
  }
}

// void k_ComputeMortonCode(const int n_threads,
//                          const glm::vec4* data,
//                          unsigned int* morton_keys,
//                          const int n,
//                          const float min_coord,
//                          const float range) {
// #pragma omp parallel for num_threads(n_threads)
//   for (auto i = 0; i < n; i++) {
//     morton_keys[i] = shared::xyz_to_morton32(data[i], min_coord, range);
//   }
// }

}  // namespace cpu

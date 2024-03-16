#include <pthread.h>
#include <unistd.h>
#include <glm/glm.hpp>
#include <iostream>

#include "shared/morton.h"

namespace cpu {

  class ThreadAffinitySetter {
public:
    static int stickThisThreadToCore(int core_id) {
        int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
        if (core_id < 0 || core_id >= num_cores)
            return EINVAL;

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);

        //pthread_t current_thread = pthread_self();

        pid_t pid = gettid();
        return sched_setaffinity(pid, sizeof(cpu_set_t), &cpuset);
    }

};


typedef struct {
  const glm::vec4* data;
  unsigned int* morton_keys;
  int start;
  int end;
  float min_coord;
  float range;
  int core_id;
} ThreadArgs;

void* computeMortonCode(void* args) {
 ThreadArgs* tData = (ThreadArgs*)args;
 // Pin thread to core 
int result = ThreadAffinitySetter::stickThisThreadToCore(tData->core_id);
 if (result == 0) {
   //std::cout << "Thread pinned to core " << tData->core_id << std::endl;
 } else {
   std::cout << "Failed to pin thread to core " << tData->core_id << std::endl;
 }
 // Perform thread tasks
  for (int i = tData->start; i < tData->end; i++) {
    tData->morton_keys[i] =
        shared::xyz_to_morton32(tData->data[i], tData->min_coord, tData->range);
  }
  return NULL;
}

void k_ComputeMortonCode(const int thread_start,
                         const int thread_end,
                         const glm::vec4* data,
                         unsigned int* morton_keys,
                         const int n,
                         const float min_coord,
                         const float range) {

  // Split data into chunks
 // std::cerr << "Thread Start " << thread_start << std::endl;
  //std::cerr << "Thread End  " << thread_end << std::endl;
  const int n_threads = (thread_end - thread_start) + 1;
  int chunkSize = n / n_threads;
  // Initialize threads
  pthread_t threads[n_threads];
  ThreadArgs threadData[n_threads];
  // Assign thread data and create threads
  for (int i = 0; i < n_threads; i++) {
    threadData[i].data = data;
    threadData[i].morton_keys = morton_keys;
    threadData[i].start = i * chunkSize;
    threadData[i].end = (i == n_threads - 1) ? n: (i + 1) * chunkSize;  // Last thread may have more work
    threadData[i].min_coord = min_coord;
    threadData[i].range = range;
    threadData[i].core_id = i / 2; // Assign every two threads to a core

    pthread_create(&threads[i], NULL, computeMortonCode, (void*)&threadData[i]);
  }
  // Join threads
  for (int i = 0; i < n_threads; i++) {
    pthread_join(threads[i], NULL);
  }
}



}  // namespace cpu

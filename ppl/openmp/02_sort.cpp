#include "openmp/kernels/02_sort.hpp"

#include <omp.h>

#include <algorithm>

constexpr int BASE_BITS = 8;
constexpr int BASE = (1 << BASE_BITS);
constexpr int MASK = (BASE - 1);

constexpr int DIGITS(const unsigned int v, const int shift) {
  return (v >> shift) & MASK;
}

// ----------------------------------------------------------------------------
// Yanwen's addition
// ----------------------------------------------------------------------------

// todo: put these into the shared temporary memory
inline void omp_lsd_radix_sort(const int n,
                               unsigned int* data,
                               unsigned int* h_buffer) {
  constexpr auto total_digits = sizeof(unsigned int) * 8;

  for (auto shift = 0; shift < total_digits; shift += BASE_BITS) {
    int bucket[BASE] = {};
    int local_bucket[BASE] = {};  // size needed in each bucket/thread

#pragma omp parallel firstprivate(local_bucket)
    {
#pragma omp for schedule(static) nowait
      for (auto i = 0; i < n; i++) {
        local_bucket[DIGITS(data[i], shift)]++;
      }
#pragma omp critical
      for (auto i = 0; i < BASE; i++) {
        bucket[i] += local_bucket[i];
      }
#pragma omp barrier
#pragma omp single
      for (auto i = 1; i < BASE; i++) {
        bucket[i] += bucket[i - 1];
      }
      const auto n_threads = omp_get_num_threads();
      const auto tid = omp_get_thread_num();
      for (auto cur_t = n_threads - 1; cur_t >= 0; cur_t--) {
        if (cur_t == tid) {
          for (auto i = 0; i < BASE; i++) {
            bucket[i] -= local_bucket[i];
            local_bucket[i] = bucket[i];
          }
        } else {
#pragma omp barrier
        }
      }
#pragma omp for schedule(static)
      for (auto i = 0; i < n; i++) {
        h_buffer[local_bucket[DIGITS(data[i], shift)]++] = data[i];
      }
    }
    // now move data
    std::swap(data, h_buffer);
  }
}

// ----------------------------------------------------------------------------
// Kernel itself
// ----------------------------------------------------------------------------

void cpu::k_Sort(const int num_threads,
                 unsigned int* u_sort,
                 unsigned int* u_sort_alt,
                 const size_t n) {
  omp_set_num_threads(num_threads);
  omp_lsd_radix_sort(n, u_sort, u_sort_alt);
}
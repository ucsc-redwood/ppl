#pragma once

#include <vector>

#include "helper.cuh"

namespace cu {

/* The allocator class */
template <typename T>
class unified_alloc {
 public:
  using value_type = T;
  using pointer = value_type *;
  using size_type = std::size_t;

  unified_alloc() noexcept = default;

  template <typename U>
  unified_alloc(unified_alloc<U> const &) noexcept {}

  auto allocate(size_type n, const void * = 0) -> value_type * {
    value_type *tmp;
    MallocManaged(&tmp, n);
    return tmp;
  }

  auto deallocate(pointer p, size_type n) -> void {
    if (p) {
      CUDA_FREE(p);
    }
  }
};

[[nodiscard]] int get_current_device() {
  auto result = int{};
  cudaGetDevice(&result);
  return result;
}

template <typename T>
struct is_unified : std::false_type {};

template <template <typename, typename> typename Outer, typename Inner>
struct is_unified<Outer<Inner, unified_alloc<Inner>>> : std::true_type {};

template <typename T>
constexpr static auto is_unified_v = is_unified<T>::value;

template <typename T>
using unified_vector = std::vector<T, unified_alloc<T>>;

template <typename T, typename = std::enable_if_t<is_unified_v<T>>>
auto prefetch(T const &container,
              cudaStream_t stream = 0,
              int device = get_current_device()) {
  using value_type = typename T::value_type;
  auto p = container.data();
  if (p) {
    CHECK_CUDA_CALL(cudaMemPrefetchAsync(
        p, container.size() * sizeof(value_type), device, stream));
  }
}

}  // namespace cu
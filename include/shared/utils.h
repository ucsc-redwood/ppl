#pragma once

#include <type_traits>

template <typename NumeratorT, typename DenominatorT>
constexpr auto DivideAndRoundUp(const NumeratorT n, const DenominatorT d)
    -> NumeratorT {
  static_assert(
      std::is_integral_v<NumeratorT> && std::is_integral_v<DenominatorT>,
      "DivideAndRoundUp is only intended for integral types.");
  // Static cast to undo integral promotion.
  return static_cast<NumeratorT>(n / d + (n % d != 0 ? 1 : 0));
}

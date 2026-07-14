#pragma once

#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/Span.hpp>
#include <cassert>
#include <cstddef>
#include <cmath>

namespace polyfem {
namespace autogen {
POLYFEM_BOTH int q_basis_count_1d(const int q);

POLYFEM_BOTH void q_basis_value_1d(const int q, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val);


}}

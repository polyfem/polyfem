#pragma once

#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/Span.hpp>
#include <cassert>
#include <cstddef>
#include <cmath>

namespace polyfem {
namespace autogen {
POLYFEM_BOTH void q_grad_basis_value_3d(const int q, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z);


}}

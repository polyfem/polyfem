#pragma once

#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/Span.hpp>
#include <cassert>
#include <cstddef>
#include <cmath>

namespace polyfem {
namespace autogen {
POLYFEM_BOTH int b_basis_count_2d(const int b);

POLYFEM_BOTH void b_basis_value_2d(const int b, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val);

POLYFEM_BOTH void b_grad_basis_value_2d(const int b, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z);

POLYFEM_BOTH int b_basis_count_3d(const int b);

POLYFEM_BOTH void b_basis_value_3d(const int b, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val);

POLYFEM_BOTH void b_grad_basis_value_3d(const int b, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z);


static const int MAX_B_BASES = 4;

}}

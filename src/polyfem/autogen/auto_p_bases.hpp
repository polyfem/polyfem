#pragma once

#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/Span.hpp>
#include <cassert>
#include <cstddef>
#include <cmath>

#include "auto_b_bases.hpp"

namespace polyfem {
namespace autogen {
POLYFEM_BOTH int p_basis_count_2d(const int p);

POLYFEM_BOTH void p_basis_value_2d(const bool bernstein, const int p, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val);

POLYFEM_BOTH void p_grad_basis_value_2d(const bool bernstein, const int p, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z);

POLYFEM_BOTH int p_basis_count_3d(const int p);

POLYFEM_BOTH void p_basis_value_3d(const bool bernstein, const int p, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> val);

POLYFEM_BOTH void p_grad_basis_value_3d(const bool bernstein, const int p, const int local_index, Span<const double> x, Span<const double> y, Span<const double> z, Span<double> grad_x, Span<double> grad_y, Span<double> grad_z);


static const int MAX_P_BASES = 4;

}}

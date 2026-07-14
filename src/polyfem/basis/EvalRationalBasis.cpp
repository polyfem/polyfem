#include <polyfem/basis/EvalRationalBasis.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>

#include <polyfem/utils/Range.hpp>

#include <cassert>

namespace polyfem::basis
{
	namespace detail
	{
		POLYFEM_BOTH Span<const double> double2span(Span<const double> values, int i)
		{
			return values.size() == 0 ? Span<const double>{} : Span<const double>{values.data() + i, 1};
		}

	} // namespace detail

	POLYFEM_BOTH int rational_basis_count(const BasisDesc &desc)
	{
		assert(desc.basis_family == BasisFamily::Rational);
		assert(desc.element_kind == ElementKind::Simplex);
		assert(desc.dim == 2);
		assert(desc.order == 2);
		return autogen::p_basis_count_2d(desc.order);
	}

	POLYFEM_BOTH void rational_basis_values_single(
		int local_basis_index,
		const BasisDesc &desc,
		const BasisStoreView &store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values)
	{
		int basis_count = rational_basis_count(desc);
		int n = x.size(); // quadrature count
		assert(values.size() == x.size());
		assert(y.size() == x.size());

		int bi = local_basis_index;
		assert(bi >= 0 && bi < basis_count);

		Span<const double> weights = slice_by_range(store.rational_weights, desc.rational_weight_range);
		assert(weights.size() == basis_count);

		for (int qi = 0; qi < n; ++qi)
		{
			auto px = detail::double2span(x, qi);
			auto py = detail::double2span(y, qi);
			auto pz = detail::double2span(z, qi);
			double denominator = 0;
			double numerator_phi = 0;
			for (int j = 0; j < basis_count; ++j)
			{
				double phi_j = 0;
				Span<double> phi_j_span(&phi_j, 1);
				autogen::p_basis_value_2d(desc.is_bernstein, desc.order, j, px, py, pz, phi_j_span);
				denominator += weights[j] * phi_j;
				if (j == bi)
					numerator_phi = phi_j;
			}
			values[qi] = weights[bi] * numerator_phi / denominator;
		}
	}

	POLYFEM_BOTH void rational_basis_values(
		const BasisDesc &desc,
		const BasisStoreView &store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values)
	{
		int basis_count = rational_basis_count(desc);
		int n = x.size(); // quadrature count
		assert(values.size() == basis_count * x.size());
		assert(y.size() == x.size());

		// loop basis
		for (int bi = 0; bi < basis_count; ++bi)
		{
			Span<double> basis_values = values.subspan(bi * n, n);
			rational_basis_values_single(bi, desc, store, x, y, z, basis_values);
		}
	}

	POLYFEM_BOTH void rational_basis_gradients_single(
		int local_basis_index,
		const BasisDesc &basis_desc,
		BasisStoreView basis_store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z)
	{
		// only 2d is supported.
		(void)grad_z;

		int basis_count = rational_basis_count(basis_desc);
		int n = x.size(); // quadrature count
		assert(grad_x.size() == x.size());
		assert(y.size() == x.size());
		assert(grad_y.size() == x.size());

		int bi = local_basis_index;
		assert(bi >= 0 && bi < basis_count);

		Span<const double> weights = slice_by_range(basis_store.rational_weights, basis_desc.rational_weight_range);
		assert(weights.size() == basis_count);

		for (int qi = 0; qi < n; ++qi)
		{
			auto px = detail::double2span(x, qi);
			auto py = detail::double2span(y, qi);
			auto pz = detail::double2span(z, qi);
			double denom = 0;
			double denom_grad_x = 0;
			double denom_grad_y = 0;
			double numer_phi = 0;
			double numer_grad_x = 0;
			double numer_grad_y = 0;

			for (int j = 0; j < basis_count; ++j)
			{
				double phi_j = 0;
				double phi_j_grad_x = 0;
				double phi_j_grad_y = 0;
				Span<double> phi_j_span(&phi_j, 1);
				Span<double> phi_j_grad_x_span(&phi_j_grad_x, 1);
				Span<double> phi_j_grad_y_span(&phi_j_grad_y, 1);
				Span<double> empty_grad_z;
				autogen::p_basis_value_2d(basis_desc.is_bernstein, basis_desc.order, j, px, py, pz, phi_j_span);
				autogen::p_grad_basis_value_2d(
					basis_desc.is_bernstein,
					basis_desc.order,
					j,
					px,
					py,
					pz,
					phi_j_grad_x_span,
					phi_j_grad_y_span,
					empty_grad_z);

				denom += weights[j] * phi_j;
				denom_grad_x += weights[j] * phi_j_grad_x;
				denom_grad_y += weights[j] * phi_j_grad_y;
				if (j == bi)
				{
					numer_phi = weights[j] * phi_j;
					numer_grad_x = weights[j] * phi_j_grad_x;
					numer_grad_y = weights[j] * phi_j_grad_y;
				}
			}

			double inv_denominator_sq = 1.0 / (denom * denom);
			grad_x[qi] = (numer_grad_x * denom - numer_phi * denom_grad_x) * inv_denominator_sq;
			grad_y[qi] = (numer_grad_y * denom - numer_phi * denom_grad_y) * inv_denominator_sq;
		}
	}

	POLYFEM_BOTH void rational_basis_gradients(
		const BasisDesc &basis_desc,
		BasisStoreView basis_store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z)
	{
		// only 2d is supported.
		(void)grad_z;

		int basis_count = rational_basis_count(basis_desc);
		int n = x.size(); // quadrature count
		assert(grad_x.size() == basis_count * x.size());
		assert(y.size() == x.size());
		assert(grad_y.size() == basis_count * x.size());

		for (int bi = 0; bi < basis_count; ++bi)
		{
			Span<double> basis_grad_x = grad_x.subspan(bi * n, n);
			Span<double> basis_grad_y = grad_y.subspan(bi * n, n);
			rational_basis_gradients_single(
				bi,
				basis_desc,
				basis_store,
				x,
				y,
				z,
				basis_grad_x,
				basis_grad_y,
				{});
		}
	}

	POLYFEM_BOTH void rational_basis_value_and_gradients(
		const BasisDesc &desc,
		BasisStoreView store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z)
	{
		rational_basis_values(desc, store, x, y, z, values);
		rational_basis_gradients(desc, store, x, y, z, grad_x, grad_y, grad_z);
	}
} // namespace polyfem::basis

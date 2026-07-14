#include <polyfem/basis/EvalLagrangeBasis.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_pyramid_bases.hpp>
#include <polyfem/autogen/auto_q_bases_1d_val.hpp>
#include <polyfem/autogen/auto_q_bases_1d_grad.hpp>
#include <polyfem/autogen/auto_q_bases_2d_val.hpp>
#include <polyfem/autogen/auto_q_bases_2d_grad.hpp>
#include <polyfem/autogen/auto_q_bases_3d_val.hpp>
#include <polyfem/autogen/auto_q_bases_3d_grad.hpp>

#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/Span.hpp>

#include <cassert>

namespace polyfem::basis
{

	POLYFEM_BOTH int lagrange_basis_count(const BasisDesc &desc)
	{
		assert(desc.basis_family == BasisFamily::Lagrange);
		if (desc.dim == 1)
		{
			return autogen::q_basis_count_1d(desc.order);
		}

		switch (desc.element_kind)
		{
		case ElementKind::Simplex:
		{
			if (desc.dim == 2)
			{
				return autogen::p_basis_count_2d(desc.order);
			}
			if (desc.dim == 3)
			{
				return autogen::p_basis_count_3d(desc.order);
			}
			assert(false);
			return 0;
		}
		case ElementKind::Quad:
		{
			assert(desc.dim == 2);
			return autogen::q_basis_count_2d(desc.order);
		}
		case ElementKind::Hex:
		{
			assert(desc.dim == 3);
			return autogen::q_basis_count_3d(desc.order);
		}
		case ElementKind::Pyramid:
		{
			assert(desc.dim == 3);
			return autogen::pyramid_basis_count_3d(desc.order);
		}
		default:
			assert(false);
		}
		return 0;
	}

	POLYFEM_BOTH void lagrange_basis_values_single(
		int local_basis_index,
		const BasisDesc &desc,
		const BasisStoreView &store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values)
	{
		// Lagrange basis doesn't need additional data from store.
		(void)store;

		int n = x.size(); // quadrature point count
		assert(desc.dim < 2 || y.size() == x.size());
		assert(desc.dim < 3 || z.size() == x.size());
		assert(desc.basis_family == BasisFamily::Lagrange);

		int id = local_basis_index;
		assert(id >= 0 && id < lagrange_basis_count(desc));

		if (desc.dim == 1)
		{
			autogen::q_basis_value_1d(desc.order, id, x, y, z, values);
			return;
		}

		switch (desc.element_kind)
		{
		case ElementKind::Simplex:
		{
			if (desc.dim == 2)
			{
				autogen::p_basis_value_2d(desc.is_bernstein, desc.order, id, x, y, z, values);
			}
			else if (desc.dim == 3)
			{
				autogen::p_basis_value_3d(desc.is_bernstein, desc.order, id, x, y, z, values);
			}
			else
			{
				assert(false);
			}
			break;
		}
		case ElementKind::Quad:
		{
			assert(desc.dim == 2);
			autogen::q_basis_value_2d(desc.order, id, x, y, z, values);
			break;
		}
		case ElementKind::Hex:
		{
			assert(desc.dim == 3);
			autogen::q_basis_value_3d(desc.order, id, x, y, z, values);
			break;
		}
		case ElementKind::Pyramid:
		{
			assert(desc.dim == 3);
			autogen::pyramid_basis_value_3d(desc.order, id, x, y, z, values);
			break;
		}
		default:
			assert(false);
		}
	}

	POLYFEM_BOTH void lagrange_basis_values(
		const BasisDesc &desc,
		const BasisStoreView &store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values)
	{
		// Lagrange basis doesn't need additional data from store.
		(void)store;

		int basis_count = lagrange_basis_count(desc);
		int n = x.size(); // quadrature point count
		assert(values.size() == basis_count * x.size());
		assert(desc.dim < 2 || y.size() == x.size());
		assert(desc.dim < 3 || z.size() == x.size());
		assert(desc.basis_family == BasisFamily::Lagrange);

		for (int i = 0; i < basis_count; ++i)
		{
			auto value_slice = values.subspan(i * n, n);
			lagrange_basis_values_single(i, desc, store, x, y, z, value_slice);
		}
	}

	POLYFEM_BOTH void lagrange_basis_gradients_single(
		int local_basis_index,
		const BasisDesc &desc,
		BasisStoreView store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z)
	{
		// Lagrange basis doesn't need additional data from store.
		(void)store;

		int n = x.size(); // quadrature count
		assert(grad_x.size() == x.size());
		assert(desc.dim < 2 || y.size() == x.size());
		assert(desc.dim < 2 || grad_y.size() == x.size());
		assert(desc.dim < 3 || z.size() == x.size());
		assert(desc.dim < 3 || grad_z.size() == x.size());
		assert(desc.basis_family == BasisFamily::Lagrange);

		int id = local_basis_index;
		assert(id >= 0 && id < lagrange_basis_count(desc));

		if (desc.dim == 1)
		{
			autogen::q_grad_basis_value_1d(
				desc.order,
				id,
				x,
				y,
				z,
				grad_x,
				{},
				{});
			return;
		}

		switch (desc.element_kind)
		{
		case ElementKind::Simplex:
		{
			if (desc.dim == 2)
			{
				autogen::p_grad_basis_value_2d(
					desc.is_bernstein,
					desc.order,
					id,
					x,
					y,
					z,
					grad_x,
					grad_y,
					{});
			}
			else if (desc.dim == 3)
			{
				autogen::p_grad_basis_value_3d(
					desc.is_bernstein,
					desc.order,
					id,
					x,
					y,
					z,
					grad_x,
					grad_y,
					grad_z);
			}
			else
			{
				assert(false);
			}
			break;
		}
		case ElementKind::Quad:
		{
			assert(desc.dim == 2);
			autogen::q_grad_basis_value_2d(
				desc.order,
				id,
				x,
				y,
				z,
				grad_x,
				grad_y,
				{});
			break;
		}
		case ElementKind::Hex:
		{
			assert(desc.dim == 3);
			autogen::q_grad_basis_value_3d(
				desc.order,
				id,
				x,
				y,
				z,
				grad_x,
				grad_y,
				grad_z);
			break;
		}
		case ElementKind::Pyramid:
		{
			assert(desc.dim == 3);
			autogen::pyramid_grad_basis_value_3d(
				desc.order,
				id,
				x,
				y,
				z,
				grad_x,
				grad_y,
				grad_z);
			break;
		}
		default:
			assert(false);
		}
	}

	POLYFEM_BOTH void lagrange_basis_gradients(
		const BasisDesc &desc,
		BasisStoreView store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z)
	{
		// Lagrange basis doesn't need additional data from store.
		(void)store;

		int basis_count = lagrange_basis_count(desc);
		int n = x.size(); // quadrature count
		assert(grad_x.size() == basis_count * x.size());
		assert(desc.dim < 2 || y.size() == x.size());
		assert(desc.dim < 2 || grad_y.size() == basis_count * x.size());
		assert(desc.dim < 3 || z.size() == x.size());
		assert(desc.dim < 3 || grad_z.size() == basis_count * x.size());
		assert(desc.basis_family == BasisFamily::Lagrange);

		for (int i = 0; i < basis_count; ++i)
		{
			auto grad_x_slice = grad_x.subspan(i * n, n);
			auto grad_y_slice = (desc.dim > 1) ? grad_y.subspan(i * n, n) : Span<double>{};
			auto grad_z_slice = (desc.dim > 2) ? grad_z.subspan(i * n, n) : Span<double>{};
			lagrange_basis_gradients_single(
				i,
				desc,
				store,
				x,
				y,
				z,
				grad_x_slice,
				grad_y_slice,
				grad_z_slice);
		}
	}

	POLYFEM_BOTH void lagrange_basis_value_and_gradients(
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
		lagrange_basis_values(desc, store, x, y, z, values);
		lagrange_basis_gradients(desc, store, x, y, z, grad_x, grad_y, grad_z);
	}

} // namespace polyfem::basis

#include <polyfem/quadrature/QuadratureOrder.hpp>

#include <algorithm>
#include <cassert>

namespace polyfem::quadrature
{
	namespace
	{
		/// Compute max polynomial order of basis gradient ∇φ w.r.t to reference coordinate.
		int grad_basis_order_max(BasisFamily family, int order)
		{
			// For simplex basis derivative subtracts order by one.
			if (family == BasisFamily::SIMPLEX)
			{
				return std::max(order - 1, 0);
			}

			// For tensor product basis gradient reduce the order of one variable while leaving
			// others unchanged. Be conservative.
			// Ex. For order 1 lagrange basis of a 2D square.
			//
			// (1,0)       (1,1)
			// o----------o
			// |          |
			// |          | v
			// |          |
			// o----------o
			// (0,0)  u    (1,0)
			//
			// φ0 = 0.25(u-1)(v-1)
			// ∂φ0/∂u = 0.25(v-1)
			// The order of u decreases while the order of v remain unchanged. Thus we estimate
			// conservatively as v still has degree 1.
			return order;
		}

		/// @brief Compute max polynomial order of det(J).
		int geom_mapping_order_max(const GeometryBasisOrderHint &hint, bool is_height_axis)
		{
			// For poly family, the order of det(J) depends on runtime solution.
			// Thus the actual order is computed on-the-fly later in PolygonQuadrature.cpp.
			if (hint.family == BasisFamily::POLY)
			{
				return 0;
			}

			int order = is_height_axis ? hint.height_order : hint.order;
			return hint.dim * grad_basis_order_max(hint.family, order);
		}

		int compute_order(
			const WeakFormOrderHint &weakform_hint,
			const BasisOrderHint &basis_hint,
			const GeometryBasisOrderHint &geometry_hint,
			bool is_height_axis)
		{
			int order =
				weakform_hint.phi_count * basis_hint.order
				+ weakform_hint.grad_phi_count * grad_basis_order_max(basis_hint.family, basis_hint.order)
				+ weakform_hint.extra_order
				+ geom_mapping_order_max(geometry_hint, is_height_axis);

			return std::max(order, 1);
		}
	} // namespace

	QuadratureOrder::QuadratureOrder(
		const WeakFormOrderHint &weakform_hint,
		const BasisOrderHint &basis_hint,
		const GeometryBasisOrderHint &geometry_hint,
		int user_override)
	{
		assert(weakform_hint.phi_count >= 0);
		assert(weakform_hint.grad_phi_count >= 0);
		assert(weakform_hint.extra_order >= 0);
		assert(basis_hint.order >= 1);
		assert(basis_hint.height_order >= 1);
		assert(geometry_hint.dim >= 1);
		assert(geometry_hint.order >= 1);
		assert(geometry_hint.height_order >= 1);

		if (user_override > 0)
		{
			order = user_override;
			height_order = user_override;
			return;
		}

		if (basis_hint.family == BasisFamily::PRISM)
		{
			order = compute_order(weakform_hint, basis_hint, geometry_hint, false);
			height_order = compute_order(weakform_hint, basis_hint, geometry_hint, true);
			return;
		}

		order = compute_order(weakform_hint, basis_hint, geometry_hint, false);
		height_order = order;
	}

} // namespace polyfem::quadrature

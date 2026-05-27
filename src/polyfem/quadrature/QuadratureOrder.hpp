#pragma once

namespace polyfem::quadrature
{
	struct WeakFormOrderHint
	{
		int phi_count = 0;
		int grad_phi_count = 0;
		int extra_order = 0;
	};

	enum class BasisFamily
	{
		SIMPLEX,
		TENSOR,
		PRISM,
		PYRAMID,
		SPLINE,
		POLY
	};

	struct BasisOrderHint
	{
		BasisFamily family = BasisFamily::SIMPLEX;
		int order = 1;
		int height_order = 1;
	};

	struct GeometryBasisOrderHint
	{
		BasisFamily family = BasisFamily::SIMPLEX;
		int dim = 1;
		int order = 1;
		int height_order = 1;
	};

	class QuadratureOrder
	{
	public:
		int order = 1;
		int height_order = 1;

		QuadratureOrder() = default;

		QuadratureOrder(
			const WeakFormOrderHint &weakform_hint,
			const BasisOrderHint &basis_hint,
			const GeometryBasisOrderHint &geometry_hint,
			const int user_override);
	};

} // namespace polyfem::quadrature

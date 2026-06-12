#pragma once

#include <string>

namespace polyfem::quadrature
{
	enum class BasisType
	{
		SIMPLEX_LAGRANGE,
		CUBE_LAGRANGE,
		PRISM_LAGRANGE,
		PYRAMID_LAGRANGE,
		SPLINE,
		POLY
	};

	/// Utility for retrieving the needed quadrature order to precisely integrate
	/// the given form on the given element basis.
	int compute_quadrature_order(const std::string &assembler, const int basis_degree, const BasisType &b_type, const int dim);
} // namespace polyfem::quadrature

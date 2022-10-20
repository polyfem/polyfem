#include "QuadQuadrature.hpp"

namespace polyfem::quadrature
{

	int Quadrature::bc_order(const int basis_degree, const int dim)
	{
		return basis_degree * 2 + 1;
	}

	int Quadrature::stiffness_poly_order(const int basis_degree, const int dim)
	{
		return (basis_degree - 1) * 2 + 1;
	}

	int Quadrature::mass_poly_order(const int basis_degree, const int dim)
	{
		return basis_degree * 2 + 1;
	}

	int Quadrature::stiffness_spline_order(const int basis_degree, const int dim)
	{
		return (basis_degree - 1) * 2 + 1; //(2 - 1) * 2 + 1
	}

	int Quadrature::mass_spline_order(const int basis_degree, const int dim)
	{
		return basis_degree * 2 + 1; // 2 * 2 + 1
	}

	int Quadrature::stiffness_order(const int basis_degree, const int dim, const bool is_hex)
	{
		if (is_hex)
			return std::max(basis_degree * 2, 1);
		else
			return std::max((basis_degree - 1) * 2, 1);
	}

	int Quadrature::mass_order(const int basis_degree, const int dim, const bool is_hex)
	{
		if (is_hex)
			return std::max(basis_degree * 2, 1);
		else
			return std::max(basis_degree * 2, 1);
	}
} // namespace polyfem::quadrature

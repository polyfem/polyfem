#include "QuadratureOrder.hpp"

#include <algorithm>

namespace polyfem::quadrature
{
	int compute_quadrature_order(const std::string &assembler, const int basis_degree, const BasisType &b_type, const int dim)
	{
		// note: minimum quadrature order is always 1
		if (assembler == "Mass")
		{
			// multiply by two since we are multiplying phi_i by phi_j
			if (b_type == BasisType::SIMPLEX_LAGRANGE || b_type == BasisType::CUBE_LAGRANGE)
				return std::max(basis_degree * 2, 1);
			else
				return basis_degree * 2 + 1;
		}
		else if (assembler == "NavierStokes")
		{
			if (b_type == BasisType::SIMPLEX_LAGRANGE)
				return std::max((basis_degree - 1) + basis_degree, 1);
			else if (b_type == BasisType::CUBE_LAGRANGE)
				return std::max(basis_degree * 2, 1);
			else
				return basis_degree * 2 + 1;
		}
		else
		{
			// subtract one since we take a derivative (lowers polynomial order by 1)
			// multiply by two since we are multiplying grad phi_i by grad phi_j
			if (b_type == BasisType::SIMPLEX_LAGRANGE)
			{
				return std::max((basis_degree - 1) * 2, 1);
			}
			else if (b_type == BasisType::CUBE_LAGRANGE || b_type == BasisType::PRISM_LAGRANGE || b_type == BasisType::PYRAMID_LAGRANGE)
			{
				// in this case we have a tensor product basis
				// this computes the quadrature order along a single axis
				// the Quadrature itself takes a tensor product of the given quadrature points
				// to form the full quadrature for the basis
				// taking a gradient leaves at least one variable whose power remains unchanged
				// thus, we don't subtract 1
				// note that this is overkill for the variable that was differentiated
				return std::max(basis_degree * 2, 1);
			}
			else
			{
				return (basis_degree - 1) * 2 + 1;
			}
		}
	}
} // namespace polyfem::quadrature

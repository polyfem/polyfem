#ifndef ELEMENT_BASES_HPP
#define ELEMENT_BASES_HPP

#include "Basis.hpp"
#include "Quadrature.hpp"


#include <vector>

namespace poly_fem
{
	///
	/// @brief      Stores the basis functions for a given element in a mesh
	///             (facet in 2d, cell in 3d).
	///
	class ElementBases
	{
	public:
		// one basis function per dof for the element
		std::vector<Basis> bases;

		// quadrature points to evaluate the basis functions inside the element
		Quadrature quadrature;

		// whether the basis functions should be evaluated in the parametric domain (FE bases),
		// or directly in the object domain (harmonic bases)
		bool has_parameterization = true;

		void eval_geom_mapping_grads(const Eigen::MatrixXd &samples, std::vector<Eigen::MatrixXd> &grads) const;
		void eval_geom_mapping(const Eigen::MatrixXd &samples, Eigen::MatrixXd &mapped) const;
	};
}

#endif

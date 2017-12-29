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
		std::vector<Basis> bases; // one basis function per dof for the element

		Quadrature quadrature; // quadrature points to evaluate the basis functions inside the element

		bool has_parameterization = true; // is it a parametric ele

		void eval_geom_mapping_grads(const Eigen::MatrixXd &samples, std::vector<Eigen::MatrixXd> &grads) const;
		void eval_geom_mapping(const Eigen::MatrixXd &samples, Eigen::MatrixXd &mapped) const;
	};
}

#endif

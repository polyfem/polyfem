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

		///
		/// @brief      { Map the sample positions in the parametric domain to
		///             the object domain (if the element has no
		///             parameterization, e.g. harmonic bases, then the
		///             parametric domain = object domain, and the mapping is
		///             identity) }
		///
		/// @param[in]  samples  { #S x dim matrix of sample positions to evaluate }
		/// @param[out] mapped   { #S x dim matrix of mapped positions }
		///
		void eval_geom_mapping(const Eigen::MatrixXd &samples, Eigen::MatrixXd &mapped) const;

		///
		/// @brief      { Evaluate the gradients of the geometric mapping
		///             defined above }
		///
		/// @param[in]  samples  { #S x dim matrix of input sample positions }
		/// @param[out] grads    { #S list of dim x dim matrices of gradients }
		///
		void eval_geom_mapping_grads(const Eigen::MatrixXd &samples, std::vector<Eigen::MatrixXd> &grads) const;
	};
}

#endif

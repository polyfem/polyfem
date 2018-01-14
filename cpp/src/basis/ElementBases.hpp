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
		typedef std::function<void(Quadrature &quadrature)> QuadratureFunction;

		// one basis function per node in the element
		std::vector<Basis> bases;

		// quadrature points to evaluate the basis functions inside the element
		void compute_quadrature(Quadrature &quadrature) const { quadrature_builder_(quadrature); }

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

		void evaluate_bases(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;
		void evaluate_grads(const Eigen::MatrixXd &uv, const int dim, Eigen::MatrixXd &grad) const;


		///
		/// @brief      {Checks if all the bases are complete }
		bool is_complete() const;

		friend std::ostream& operator<< (std::ostream& os, const ElementBases &obj)
		{
			for(std::size_t i = 0; i < obj.bases.size(); ++i)
				os << "local base "<<i <<":\n" << obj.bases[i] <<"\n";

			return os;
		}

		void set_quadrature(const QuadratureFunction &fun) { quadrature_builder_ = fun; }

	private:
		QuadratureFunction quadrature_builder_;
	};
}

#endif

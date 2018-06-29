#ifndef ELEMENT_BASES_HPP
#define ELEMENT_BASES_HPP

#include <polyfem/Basis.hpp>
#include <polyfem/Quadrature.hpp>
#include <polyfem/Mesh.hpp>


#include <vector>

namespace polyfem
{
	///
	/// @brief      Stores the basis functions for a given element in a mesh
	///             (facet in 2d, cell in 3d).
	///
	class ElementBases
	{
	public:
		typedef std::function<Eigen::VectorXi(const int local_index, const Mesh &mesh)> LocalNodeFromPrimitiveFunc;
		typedef std::function<void(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)> EvalBasesFunc;
		typedef std::function<void(const Eigen::MatrixXd &uv, const int dim, Eigen::MatrixXd &grad)> EvalGradsFunc;
		typedef std::function<void(Quadrature &quadrature)> QuadratureFunction;

		// one basis function per node in the element
		std::vector<Basis> bases;

		// quadrature points to evaluate the basis functions inside the element
		void compute_quadrature(Quadrature &quadrature) const { quadrature_builder_(quadrature); }
		Eigen::VectorXi local_nodes_for_primitive(const int local_index, const Mesh &mesh) const { return local_node_from_primitive_(local_index, mesh); }

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

		///
		/// @brief      { Checks if all the bases are complete }
		bool is_complete() const;

		friend std::ostream& operator<< (std::ostream& os, const ElementBases &obj)
		{
			for(std::size_t i = 0; i < obj.bases.size(); ++i)
				os << "local base "<<i <<":\n" << obj.bases[i] <<"\n";

			return os;
		}

		void set_quadrature(const QuadratureFunction &fun) { quadrature_builder_ = fun; }

		void evaluate_bases(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const { if (eval_bases_func_) { eval_bases_func_(uv, val); } else { evaluate_bases_default(uv, val); } }

		void evaluate_grads(const Eigen::MatrixXd &uv, const int dim, Eigen::MatrixXd &grad) const { if (eval_grads_func_) { eval_grads_func_(uv, dim, grad); } else { evaluate_grads_default(uv, dim, grad); } }

		void set_bases_func(EvalBasesFunc fun) { eval_bases_func_ = fun; }

		void set_grads_func(EvalGradsFunc fun) { eval_grads_func_ = fun; }

		void set_local_node_from_primitive_func(LocalNodeFromPrimitiveFunc fun) { local_node_from_primitive_ = fun; }

	private:
		void evaluate_bases_default(const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;

		void evaluate_grads_default(const Eigen::MatrixXd &uv, const int dim, Eigen::MatrixXd &grad) const;

	private:
		EvalBasesFunc eval_bases_func_;
		EvalGradsFunc eval_grads_func_;
		QuadratureFunction quadrature_builder_;

		LocalNodeFromPrimitiveFunc local_node_from_primitive_;
	};
}

#endif

#pragma once

#include <polyfem/basis/Basis.hpp>
#include <polyfem/quadrature/Quadrature.hpp>
#include <polyfem/mesh/Mesh.hpp>

#include <polyfem/assembler/AssemblyValues.hpp>

#include <vector>

namespace polyfem::basis
{
	/// @brief Stores the basis functions for a given element in a mesh.
	class ElementBases
	{
	public:
		typedef std::function<Eigen::VectorXi(const int local_index, const mesh::Mesh &mesh)> LocalNodeFromPrimitiveFunc;

		/// @brief Function type that evaluates bases at given points and saves them in basis_values.
		typedef std::function<void(const Eigen::MatrixXd &uv, std::vector<assembler::AssemblyValues> &basis_values)> EvalBasesFunc;

		/// @brief Function type that computes quadrature points and saves them in quadrature.
		typedef std::function<void(quadrature::Quadrature &quadrature)> QuadratureFunction;

		/// @brief Assemble the global nodal positions of the bases.
		Eigen::MatrixXd nodes() const;

		/// @brief Quadrature points to evaluate the basis functions inside the element.
		void compute_quadrature(quadrature::Quadrature &quadrature) const { quadrature_builder_(quadrature); }
		void compute_mass_quadrature(quadrature::Quadrature &quadrature) const { mass_quadrature_builder_(quadrature); }
		Eigen::VectorXi local_nodes_for_primitive(const int local_index, const mesh::Mesh &mesh) const { return local_node_from_primitive_(local_index, mesh); }

		/// @brief Whether the basis functions should be evaluated in the parametric domain (FE bases),
		/// or directly in the object domain (harmonic bases).
		bool has_parameterization = true;

		/// @brief Map the sample positions in the parametric domain to the object domain.
		/// If the element has no parameterization, e.g. harmonic bases, then
		/// the parametric domain = object domain, and the mapping is identity.
		/// @param[in] samples #S x dim matrix of sample positions to evaluate.
		/// @param[out] mapped #S x dim matrix of mapped positions.
		void eval_geom_mapping(const Eigen::MatrixXd &samples, Eigen::MatrixXd &mapped) const;

		/// @brief Evaluate the gradients of the geometric mapping defined above.
		/// @param[in] samples #S x dim matrix of input sample positions.
		/// @param[out] grads #S list of dim x dim matrices of gradients.
		void eval_geom_mapping_grads(const Eigen::MatrixXd &samples, std::vector<Eigen::MatrixXd> &grads) const;

		/// @brief Checks if all the bases are complete.
		bool is_complete() const;

		/// @brief Set the quadrature rule to use.
		void set_quadrature(const QuadratureFunction &fun) { quadrature_builder_ = fun; }
		/// @brief Set the mass quadrature rule to use.
		void set_mass_quadrature(const QuadratureFunction &fun) { mass_quadrature_builder_ = fun; }

		/// @brief Evaluate stored bases at given points on the reference element.
		/// Results are saved in basis_values.
		void evaluate_bases(
			const Eigen::MatrixXd &uv,
			std::vector<assembler::AssemblyValues> &basis_values) const
		{
			if (eval_bases_func_)
			{
				eval_bases_func_(uv, basis_values);
			}
			else
			{
				evaluate_bases_default(uv, basis_values);
			}
		}

		/// @brief Evaluate gradient of stored bases at given points on the reference element.
		/// Results are saved in basis_values.
		void evaluate_grads(
			const Eigen::MatrixXd &uv,
			std::vector<assembler::AssemblyValues> &basis_values) const
		{
			if (eval_grads_func_)
			{
				eval_grads_func_(uv, basis_values);
			}
			else
			{
				evaluate_grads_default(uv, basis_values);
			}
		}

		/// @brief Set the function to evaluate the bases.
		void set_bases_func(EvalBasesFunc fun) { eval_bases_func_ = fun; }
		/// @brief Set the function to evaluate the bases gradients.
		void set_grads_func(EvalBasesFunc fun) { eval_grads_func_ = fun; }

		/// @brief Set mapping from local nodes to global nodes.
		void set_local_node_from_primitive_func(LocalNodeFromPrimitiveFunc fun) { local_node_from_primitive_ = fun; }

		/// @brief Print the bases to the output stream.
		friend std::ostream &operator<<(std::ostream &os, const ElementBases &obj);

		// --- Public members ---------------------------------------------

		/// @brief One basis function per node in the element.
		std::vector<Basis> bases;

	private:
		/// @brief Default to simply calling the Basis evaluation functions.
		void evaluate_bases_default(const Eigen::MatrixXd &uv, std::vector<assembler::AssemblyValues> &basis_values) const;

		/// @brief Default to simply calling the Basis gradient functions.
		void evaluate_grads_default(const Eigen::MatrixXd &uv, std::vector<assembler::AssemblyValues> &basis_values) const;

		/// @brief Function to evaluate the basis functions at given points.
		EvalBasesFunc eval_bases_func_;

		/// @brief Function to evaluate the basis gradients at given points.
		EvalBasesFunc eval_grads_func_;

		/// @brief Function to compute quadrature points.
		QuadratureFunction quadrature_builder_;

		/// @brief Function to compute mass quadrature points.
		QuadratureFunction mass_quadrature_builder_;

		/// @brief Function to map local nodes to global nodes.
		LocalNodeFromPrimitiveFunc local_node_from_primitive_;
	};
} // namespace polyfem::basis

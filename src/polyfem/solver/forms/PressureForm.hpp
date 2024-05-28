#pragma once

#include "Form.hpp"

#include <polyfem/assembler/PressureAssembler.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	/// @brief Form representing body forces
	class PressureForm : public Form
	{
	public:
		/// @brief Construct a new Body Form object
		/// @param state Reference to the simulation state
		/// @param pressure_assembler Reference to the pressure assembler
		PressureForm(const int ndof,
					 const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
					 const std::unordered_map<int, std::vector<mesh::LocalBoundary>> &local_pressure_cavity,
					 const std::vector<int> &dirichlet_nodes,
					 const int n_boundary_samples,
					 const assembler::PressureAssembler &pressure_assembler,
					 const bool is_time_dependent);

		std::string name() const override { return "pressure"; }

	protected:
		/// @brief Compute the value of the body force form
		/// @param x Current solution
		/// @return Value of the body force form
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Vector containing the current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	public:
		/// @brief Update time dependent quantities
		/// @param t New time
		/// @param x Solution at time t
		void update_quantities(const double t, const Eigen::VectorXd &x) override;

		/// @brief Compute the derivative of the force wrt vertex positions, then multiply the resulting matrix with adjoint_sol.
		/// @param[in] n_verts Number of vertices
		/// @param[in] x Current solution
		/// @param[in] adjoint Current adjoint solution
		/// @param[out] term Derivative of force multiplied by the adjoint
		void force_shape_derivative(
			const int n_verts,
			const double t,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &term);

		/// @brief Compute the derivative of the force wrt vertex positions, then multiply the resulting matrix with adjoint_sol.
		/// @param[in] n_verts Number of vertices
		/// @param[in] x Current solution
		/// @param[in] adjoint Current adjoint solution
		/// @param[out] term Derivative of force multiplied by the adjoint
		double force_pressure_derivative(
			const int n_verts,
			const double t,
			const int pressure_boundary_id,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &adjoint);

	private:
		double t_;       ///< Current time
		const int ndof_; ///< Number of degrees of freedom

		const std::vector<mesh::LocalBoundary> &local_pressure_boundary_;
		const std::unordered_map<int, std::vector<mesh::LocalBoundary>> &local_pressure_cavity_;
		const std::vector<int> &dirichlet_nodes_;
		const int n_boundary_samples_;

		const assembler::PressureAssembler &pressure_assembler_; ///< Reference to the pressure assembler
	};
} // namespace polyfem::solver

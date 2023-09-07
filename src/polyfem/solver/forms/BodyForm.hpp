#pragma once

#include "Form.hpp"

#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	/// @brief Form representing body forces
	class BodyForm : public Form
	{
	public:
		/// @brief Construct a new Body Form object
		/// @param state Reference to the simulation state
		/// @param rhs_assembler Reference to the right hand side assembler
		/// @param apply_DBC If true, set the Dirichlet boundary conditions in the RHS
		BodyForm(const int ndof,
				 const int n_pressure_bases,
				 const std::vector<int> &boundary_nodes,
				 const std::vector<mesh::LocalBoundary> &local_boundary,
				 const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
				 const int n_boundary_samples,
				 const Eigen::MatrixXd &rhs,
				 const assembler::RhsAssembler &rhs_assembler,
				 const assembler::Density &density,
				 const bool apply_DBC,
				 const bool is_formulation_mixed,
				 const bool is_time_dependent);

		std::string name() const override { return "body"; }

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

		bool get_apply_DBC() { return apply_DBC_; }
		void set_apply_DBC(const Eigen::VectorXd &x, const bool val) override
		{
			if (val != apply_DBC_)
			{
				apply_DBC_ = val;
				update_current_rhs(x);
			}
		}

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

		void hessian_wrt_u_prev(
			const Eigen::VectorXd &u_prev,
			const double t,
			StiffnessMatrix &hessian) const;

	private:
		double t_;       ///< Current time
		const int ndof_; ///< Number of degrees of freedom
		const int n_pressure_bases_;

		const std::vector<int> &boundary_nodes_;
		const std::vector<mesh::LocalBoundary> &local_boundary_;
		const std::vector<mesh::LocalBoundary> &local_neumann_boundary_;
		const int n_boundary_samples_;

		const Eigen::MatrixXd &rhs_;                   ///< static RHS for the current time
		const assembler::RhsAssembler &rhs_assembler_; ///< Reference to the RHS assembler
		const assembler::Density &density_;

		bool apply_DBC_;            ///< If true, set the Dirichlet boundary conditions in the RHS
		bool is_formulation_mixed_; ///< True if the formulation is mixed

		Eigen::MatrixXd current_rhs_; ///< Cached RHS for the current time

		/// @brief Update current_rhs
		void update_current_rhs(const Eigen::VectorXd &x);
	};
} // namespace polyfem::solver

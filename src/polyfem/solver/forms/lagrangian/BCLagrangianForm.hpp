#pragma once

#include "AugmentedLagrangianForm.hpp"
#include <polyfem/assembler/RhsAssembler.hpp>

#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>

namespace polyfem::utils
{
	class PeriodicBoundary;
} // namespace polyfem::utils

namespace polyfem::solver
{
	/// @brief Form of the augmented lagrangian for bc constraints
	class BCLagrangianForm : public AugmentedLagrangianForm
	{
	public:
		/// @brief Construct a new BCLagrangianForm object with a time dependent Dirichlet boundary
		/// @param ndof Number of degrees of freedom
		/// @param boundary_nodes DoFs that are part of the Dirichlet boundary
		/// @param local_boundary
		/// @param local_neumann_boundary
		/// @param n_boundary_samples
		/// @param mass Mass matrix
		/// @param rhs_assembler Right hand side assembler
		/// @param obstacle_ndof Obstacle's number of DOF
		/// @param is_time_dependent Whether the problem is time dependent
		/// @param t Current time
		/// @param periodic_bc Periodic boundary conditions
		BCLagrangianForm(const int ndof,
						 const std::vector<int> &boundary_nodes,
						 const std::vector<mesh::LocalBoundary> &local_boundary,
						 const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
						 const int n_boundary_samples,
						 const StiffnessMatrix &mass,
						 const assembler::RhsAssembler &rhs_assembler,
						 const size_t obstacle_ndof,
						 const bool is_time_dependent,
						 const double t,
						 const std::shared_ptr<utils::PeriodicBoundary> &periodic_bc = nullptr);

		std::string name() const override
		{
			return "bc-alagrangian";
		}

		/// @brief Construct a new BCLagrangianForm object with a fixed Dirichlet boundary
		/// @param ndof Number of degrees of freedom
		/// @param boundary_nodes DoFs that are part of the Dirichlet boundary
		/// @param mass Mass matrix
		/// @param obstacle_ndof Obstacle's number of DOF
		/// @param target_x Target values for the boundary DoFs
		BCLagrangianForm(const int ndof,
						 const std::vector<int> &boundary_nodes,
						 const StiffnessMatrix &mass,
						 const size_t obstacle_ndof,
						 const Eigen::MatrixXd &target_x);

		/// @brief Compute the value of the form
		/// @param x Current solution
		/// @return Computed value
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
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

		void update_lagrangian(const Eigen::VectorXd &x, const double k_al) override;

		StiffnessMatrix &mask()
		{
			return mask_;
		}
		const StiffnessMatrix &mask() const { return mask_; }
		Eigen::VectorXd target(const Eigen::VectorXd &) const override { return target_x_; }

		double compute_error(const Eigen::VectorXd &x) const override;

	private:
		const std::vector<int> &boundary_nodes_;
		const std::vector<mesh::LocalBoundary> *local_boundary_;
		const std::vector<mesh::LocalBoundary> *local_neumann_boundary_;
		const int n_boundary_samples_;

		const assembler::RhsAssembler *rhs_assembler_; ///< Reference to the RHS assembler
		const bool is_time_dependent_;

		StiffnessMatrix masked_lumped_mass_sqrt_; ///< sqrt mass matrix masked by the AL dofs
		StiffnessMatrix masked_lumped_mass_;      ///< mass matrix masked by the AL dofs
		StiffnessMatrix mask_;                    ///< identity matrix masked by the AL dofs

		Eigen::MatrixXd target_x_; ///< actually a vector with the same size as x with target nodal positions

		/// @brief Initialize the masked lumped mass matrix
		/// @param ndof Number of degrees of freedom
		/// @param mass Mass matrix
		/// @param obstacle_ndof Obstacle's number of DOF
		void init_masked_lumped_mass(
			const int ndof,
			const StiffnessMatrix &mass,
			const size_t obstacle_ndof);

		/// @brief Update target x to the Dirichlet boundary values at time t
		/// @param t Current time
		void update_target(const double t);
	};
} // namespace polyfem::solver

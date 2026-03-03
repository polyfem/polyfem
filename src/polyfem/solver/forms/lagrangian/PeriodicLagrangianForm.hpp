#pragma once

#include "AugmentedLagrangianForm.hpp"
#include <polyfem/assembler/PeriodicBoundary.hpp>

namespace polyfem::solver
{
	/// @brief Form of the augmented lagrangian for bc constraints
	class PeriodicLagrangianForm : public AugmentedLagrangianForm
	{
	public:
		/// @brief Construct a new PeriodicLagrangianForm object
		/// @param ndof Number of degrees of freedom
		/// @param boundary_nodes DoFs that are part of the Periodic boundary
		/// @param n_boundary_samples
		/// @param periodic_bc Periodic boundary conditions
		PeriodicLagrangianForm(const int ndof,
							   const std::shared_ptr<utils::PeriodicBoundary> &periodic_bc);

		std::string name() const override
		{
			return "periodic-alagrangian";
		}

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
		void update_quantities(const double t, const Eigen::VectorXd &x) override {}

		void update_lagrangian(const Eigen::VectorXd &x, const double k_al) override;

		double compute_error(const Eigen::VectorXd &x) const override;

		virtual bool can_project() const override;
		virtual void project_gradient(Eigen::VectorXd &grad) const override;
		virtual void project_hessian(StiffnessMatrix &hessian) const override;

	private:
		const std::shared_ptr<utils::PeriodicBoundary> periodic_bc_;
		const int n_dofs_;
		Eigen::VectorXi constraints_;     ///< Constraints
		Eigen::VectorXi not_constraints_; ///< Not Constraints
	};
} // namespace polyfem::solver

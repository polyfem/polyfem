#pragma once

#include "LagrangianPenaltyForm.hpp"

namespace polyfem::solver
{
	/// @brief Form of the penalty in augmented lagrangian
	class GenericPenaltyForm : public LagrangianPenaltyForm
	{
	public:
		/// @brief Construct a new GenericPenaltyForm object for the constraints Ax = b
		/// @param A Constraints matrix
		/// @param b Constraints value
		GenericPenaltyForm(const StiffnessMatrix &A,
						   const Eigen::VectorXd &b);

		std::string name() const override { return "generic-penalty"; }

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
		double compute_error(const Eigen::VectorXd &x) const override;

	private:
		const StiffnessMatrix A;
		const Eigen::VectorXd b;
	};
} // namespace polyfem::solver

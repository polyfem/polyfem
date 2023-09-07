
#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{

	class L2ProjectionForm : public polyfem::solver::Form
	{
	public:
		L2ProjectionForm(
			const StiffnessMatrix &M,
			const StiffnessMatrix &A,
			const Eigen::VectorXd &y);

		std::string name() const override { return "L2_projection"; }

	protected:
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

		StiffnessMatrix M_;
		Eigen::VectorXd rhs_;
	};

} // namespace polyfem::solver
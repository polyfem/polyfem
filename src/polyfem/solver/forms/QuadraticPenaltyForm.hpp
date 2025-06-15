#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	/// @brief Form for quadratic soft constraints
	class QuadraticPenaltyForm : public Form
	{
	public:
		/// @brief Construct a new QuadraticPenaltyForm object for the constraints Ax = b
		/// @param A Constraints matrix
		/// @param b Constraints value
		/// @param weigth weigth of the penalty
		QuadraticPenaltyForm(const StiffnessMatrix &A,
							 const Eigen::MatrixXd &b,
							 const double weight);

		std::string name() const override { return "quadratic-penalty"; }

		double weight() const override { return weight_ * penalty_weight_; }

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

	private:
		StiffnessMatrix A_; ///< Constraints matrix
		Eigen::MatrixXd b_; ///< Constraints value

		StiffnessMatrix AtA_;
		Eigen::VectorXd Atb_;

		const double penalty_weight_;
	};
} // namespace polyfem::solver
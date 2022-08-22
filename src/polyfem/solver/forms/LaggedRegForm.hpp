#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	/// @brief Tikonov regularization form between x and x_lagged
	class LaggedRegForm : public Form
	{
	public:
		/// @brief Construct a new Lagged Regularization Form object
		/// @param weight Weight of the regularization term
		LaggedRegForm(const double weight_);

		/// @brief Compute the value of the form
		/// @param x Current solution
		/// @return Computed value
		double value(const Eigen::VectorXd &x) override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

		/// @brief Update lagged fields
		/// @param x Current solution
		void update_lagging(const Eigen::VectorXd &x) override;

	private:
		Eigen::VectorXd x_lagged_; ///< The full variables from the previous lagging solve.
		const double weight_;      ///< Weight of the regularization term
	};
} // namespace polyfem::solver

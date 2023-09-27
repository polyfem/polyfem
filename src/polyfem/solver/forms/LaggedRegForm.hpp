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
		LaggedRegForm(const int n_lagging_iters);

		std::string name() const override { return "lagged-regularization"; }

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

	public:
		/// @brief Initialize lagged fields
		/// @param x Current solution
		void init_lagging(const Eigen::VectorXd &x) override;

		/// @brief Update lagged fields
		/// @param x Current solution
		/// @return True if the lagged fields have been updated
		void update_lagging(const Eigen::VectorXd &x, const int iter_num) override;

		/// @brief Does this form require lagging?
		/// @return True if the form requires lagging
		bool uses_lagging() const override { return true; }

	private:
		int n_lagging_iters_;      ///< Number of iterations to lag for
		Eigen::VectorXd x_lagged_; ///< The full variables from the previous lagging solve.
	};
} // namespace polyfem::solver

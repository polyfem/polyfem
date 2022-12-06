#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	/// @brief Linear form of the form cᵀx where c is a constant vector of coefficients.
	class LinearForm : public Form
	{
	public:
		/// @brief Construct a new Linear Form object
		/// @param coeffs Coefficients of the linear form
		LinearForm(const Eigen::VectorXd &coeffs);

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

	private:
		Eigen::VectorXd coeffs_; ///< Coefficients of the linear form.
	};
} // namespace polyfem::solver

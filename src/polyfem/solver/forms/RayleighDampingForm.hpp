#pragma once

#include "Form.hpp"

#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	/// @brief Tikonov regularization form between x and x_lagged
	class RayleighDampingForm : public Form
	{
	public:
		/// @brief Construct a new Lagged Regularization Form object
		RayleighDampingForm(
			ElasticForm &elastic_form,
			const time_integrator::ImplicitTimeIntegrator &time_integrator,
			const double stiffness_ratio,
			const int n_lagging_iters);

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
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

	public:
		/// @brief Initialize lagged fields
		/// @param x Current solution
		void init_lagging(const Eigen::VectorXd &x) override;

		/// @brief Update lagged fields
		/// @param x Current solution
		/// @return True if the lagged fields have been updated
		void update_lagging(const Eigen::VectorXd &x, const int iter_num) override;

		/// @brief Get the maximum number of lagging iteration allowable.
		int max_lagging_iterations() const override { return n_lagging_iters_; }

		/// @brief Does this form require lagging?
		/// @return True if the form requires lagging
		bool uses_lagging() const override { return true; }

	private:
		double stiffness() const { return 0.75 * stiffness_ratio_ * std::pow(time_integrator_.dt(), 3); }

		// TODO: Make this const by making ElasticForm::second_derivative_unweighted const
		ElasticForm &elastic_form_;                                      ///< Reference to the elastic form
		const time_integrator::ImplicitTimeIntegrator &time_integrator_; ///< Reference to the time integrator
		double stiffness_ratio_;                                         ///< Damping stiffness coefficient
		int n_lagging_iters_;                                            ///< Number of iterations to lag for

		StiffnessMatrix lagged_stiffness_matrix_; ///< The lagged stiffness matrix
	};
} // namespace polyfem::solver

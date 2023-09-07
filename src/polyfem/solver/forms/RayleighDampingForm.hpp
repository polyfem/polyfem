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
			const Form &form_to_damp,
			const time_integrator::ImplicitTimeIntegrator &time_integrator,
			const bool use_stiffness_as_ratio,
			const double stiffness,
			const int n_lagging_iters);

		static std::shared_ptr<RayleighDampingForm> create(
			const json &params,
			const std::unordered_map<std::string, std::shared_ptr<Form>> &forms,
			const time_integrator::ImplicitTimeIntegrator &time_integrator);

		std::string name() const override { return "rayleigh-damping"; }

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

		/// @brief Get the maximum number of lagging iteration allowable.
		int max_lagging_iterations() const override { return n_lagging_iters_; }

		/// @brief Does this form require lagging?
		/// @return True if the form requires lagging
		bool uses_lagging() const override { return true; }

		/// @brief Get the stiffness of the form
		double stiffness() const;

	private:
		const Form &form_to_damp_;                                       ///< Reference to the form we are damping
		const time_integrator::ImplicitTimeIntegrator &time_integrator_; ///< Reference to the time integrator
		const bool use_stiffness_as_ratio_;                              ///< Whether to use the stiffness ratio or the stiffness value
		const double stiffness_;                                         ///< Damping stiffness coefficient
		const int n_lagging_iters_;                                      ///< Number of iterations to lag for

		StiffnessMatrix lagged_stiffness_matrix_; ///< The lagged stiffness matrix
	};
} // namespace polyfem::solver

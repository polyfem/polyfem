#pragma once

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

namespace polyfem::time_integrator
{
	/// @brief Backward Differential Formulas
	/// \f[
	/// 	x^{t+1} = \left(\sum_{i=0}^{n-1} \alpha_{i} x^{t-i}\right)+ \Delta t \beta v^{t+1}\newline
	/// 	v^{t+1} = \left(\sum_{i=0}^{n-1} \alpha_{i} v^{t-i}\right)+ \Delta t \beta a^{t+1}
	/// \f]
	/// @see https://en.wikipedia.org/wiki/Backward_differentiation_formula
	class BDF : public ImplicitTimeIntegrator
	{
	public:
		BDF(const int order = 1);

		/// @brief Set the number of steps parameters from a json object.
		/// @param params json containing `{"steps": 1}`
		void set_parameters(const json &params) override;

		/// @brief Update the time integration quantities (i.e., \f$x\f$, \f$v\f$, and \f$a\f$).
		/// @param x new solution vector
		void update_quantities(const Eigen::VectorXd &x) override;

		/// @brief Compute the predicted solution to be used in the inertia term \f$(x-\tilde{x})^TM(x-\tilde{x})\f$.
		/// \f[
		/// 	\tilde{x} = \left(\sum_{i=0}^{n-1} \alpha_i x^{t-i}\right) + \beta \Delta t \left(\sum_{i=0}^{n-1} \alpha_i v^{t-i}\right)
		/// \f]
		/// @return value for \f$\tilde{x}\f$
		Eigen::VectorXd x_tilde() const override;

		/// @brief Compute the current velocity given the current solution and using the stored previous solution(s).
		/// \f[
		/// 	v = \frac{x - \sum_{i=0}^{n-1} \alpha_i x^{t-i}}{\beta \Delta t}
		/// \f]
		/// @param x current solution vector
		/// @return value for \f$v\f$
		Eigen::VectorXd compute_velocity(const Eigen::VectorXd &x) const override;

		/// @brief Compute the current acceleration given the current velocity and using the stored previous velocity(s).
		/// \f[
		/// 	a = \frac{v - \sum_{i=0}^{n-1} \alpha_i v^{t-i}}{\beta \Delta t}
		/// \f]
		/// @param v current velocity
		/// @return value for \f$a\f$
		Eigen::VectorXd compute_acceleration(const Eigen::VectorXd &v) const override;

		/// @brief Compute the acceleration scaling used to scale forces when integrating a second order ODE.
		/// \f[
		/// 	\beta^2 \Delta t^2
		/// \f]
		double acceleration_scaling() const override;

		/// @brief Compute the derivative of the velocity with respect to the solution.
		/// \f[
		/// 	\frac{\partial v}{\partial x} = \frac{1}{\beta \Delta t}
		/// \f]
		/// \f[
		/// 	\frac{\partial v}{\partial x^{t-i}} = \frac{-\alpha_i}{\beta \Delta t}
		/// \f]
		/// \f[
		/// 	\frac{\partial v}{\partial x^{t-n}} = 0
		/// \f]
		/// @param prev_ti index of the previous solution to use (0 -> current; 1 -> previous; 2 -> second previous; etc.)
		double dv_dx(const unsigned prev_ti = 0) const override;

		/// @brief Compute \f$\beta\Delta t\f$
		double beta_dt() const;

		/// @brief Compute the weighted sum of the previous solutions.
		/// \f[
		/// 	\sum_{i=0}^{n-1} \alpha_i x^{t-i}
		/// \f]
		Eigen::VectorXd weighted_sum_x_prevs() const;

		/// @brief Compute the weighted sum of the previous velocities.
		/// \f[
		/// 	\sum_{i=0}^{n-1} \alpha_i v^{t-i}
		/// \f]
		Eigen::VectorXd weighted_sum_v_prevs() const;

		/// @brief Retrieve the alphas used for BDF with `i` steps.
		/// @param i number of steps
		/// @see https://en.wikipedia.org/wiki/Backward_differentiation_formula#General_formula
		static const std::vector<double> &alphas(const int i);

		/// @brief Retrieve the value of beta used for BDF with `i` steps.
		/// @param i number of steps
		/// @see https://en.wikipedia.org/wiki/Backward_differentiation_formula#General_formula
		static double betas(const int i);

	protected:
		/// @brief Get the maximum number of steps to use for integration.
		int max_steps() const override { return max_steps_; }

		/// @brief The maximum number of steps to use for integration.
		int max_steps_ = 1;
	};
} // namespace polyfem::time_integrator

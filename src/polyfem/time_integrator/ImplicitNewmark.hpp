#pragma once

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

namespace polyfem::time_integrator
{
	/// Implicit Newmark-beta method.
	/// \f[
	/// 	x^{t+1} = x^t + \Delta t v^t + \frac{\Delta t^2}{2}((1-2\beta)a^t + 2 \beta a^{t+1})\newline
	/// 	v^{t+1} = v^t + (1-\gamma)\Delta ta^t + \gamma \Delta ta^{t+1}
	/// \f]
	/// @see https://en.wikipedia.org/wiki/Newmark-beta_method
	class ImplicitNewmark : public ImplicitTimeIntegrator
	{
	public:
		ImplicitNewmark() {}

		/// @brief Set the `gamma` and `beta` parameters from a json object.
		/// @param params json containing `{"gamma": 0.5, "beta": 0.25}`
		void set_parameters(const json &params) override;

		/// @brief Update the time integration quantities (i.e., \f$x\f$, \f$v\f$, and \f$a\f$).
		/// @param x new solution vector
		void update_quantities(const Eigen::VectorXd &x) override;

		/// @brief Compute the predicted solution to be used in the inertia term \f$(x-\tilde{x})^TM(x-\tilde{x})\f$.
		/// \f[
		/// 	\tilde{x} = x^t + \Delta t (v^t + (0.5 - \beta) \Delta t a^t)
		/// \f]
		/// @return value for \f$\tilde{x}\f$
		Eigen::VectorXd x_tilde() const override;

		/// @brief Compute the current velocity given the current solution and using the stored previous solution(s).
		/// \f[
		/// 	a^{t+1} = \frac{x - (x^t + \Delta t v^t + \Delta t^2 (0.5 - \beta) a^t)}{\beta \Delta t^2}\newline
		/// 	v = v^t + \Delta t (1 - \gamma) a^t + \Delta t \gamma a^{t+1}
		///       = (\gamma / \beta) / \Delta t (x - x^t) + (1 - \gamma / \beta) v^t + (1 - \gamma / \beta / 2) \Delta t a^t
		/// \f]
		/// @param x current solution vector
		/// @return value for \f$v\f$
		Eigen::VectorXd compute_velocity(const Eigen::VectorXd &x) const override;

		/// @brief Compute the current acceleration given the current velocity and using the stored previous velocity(s).
		/// \f[
		/// 	a = \frac{v - v^t - (1 - \gamma) \Delta t a^t)}{\gamma \Delta t}
		/// \f]
		/// @param v current velocity
		/// @return value for \f$a\f$
		Eigen::VectorXd compute_acceleration(const Eigen::VectorXd &v) const override;

		/// @brief Compute the acceleration scaling used to scale forces when integrating a second order ODE.
		/// \f[
		/// 	\beta \Delta t^2
		/// \f]
		double acceleration_scaling() const override;

		/// @brief Compute the derivative of the velocity with respect to the solution.
		/// \f[
		/// 	\frac{\partial v}{\partial x} = \frac{\gamma}{\beta\Delta t}
		/// \f]
		/// \f[
		/// 	\frac{\partial v}{\partial x^t} = ?
		/// \f]
		/// \f[
		/// 	\frac{\partial v}{\partial x^{t-1}} = ?
		/// \f]
		/// @param prev_ti index of the previous solution to use (0 -> current; 1 -> previous; 2 -> second previous; etc.)
		double dv_dx(const unsigned prev_ti = 0) const override;

		double da_dx(const unsigned prev_ti = 0) const;

		/// @brief \f$\beta\f$ parameter for blending accelerations in the solution update.
		double beta() const { return beta_; }
		/// @brief \f$\gamma\f$ parameter for blending accelerations in the velocity update.
		double gamma() const { return gamma_; }

	protected:
		/// @brief \f$\beta\f$ parameter for blending accelerations in the solution update.
		double beta_ = 0.25;
		/// @brief \f$\gamma\f$ parameter for blending accelerations in the velocity update.
		double gamma_ = 0.5;
	};
} // namespace polyfem::time_integrator

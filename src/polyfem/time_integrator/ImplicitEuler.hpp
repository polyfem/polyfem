#pragma once

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

namespace polyfem::time_integrator
{
	/// Implicit Euler time integrator of a second order ODE (equivently a system of coupled first order ODEs).
	/// \f[
	/// 	x^{t+1} = x^t + \Delta t v^{t+1}\newline
	/// 	v^{t+1} = v^t + \Delta t a^{t+1}
	/// \f]
	/// @see https://en.wikipedia.org/wiki/Backward_Euler_method
	class ImplicitEuler : public ImplicitTimeIntegrator
	{
	public:
		ImplicitEuler() {}

		/// @brief Update the time integration quantities (i.e., \f$x\f$, \f$v\f$, and \f$a\f$).
		/// \f[
		/// 	v^{t+1} = \frac{1}{\Delta t} (x - x^t)\newline
		/// 	a^{t+1} = \frac{1}{\Delta t} (v - v^t)
		/// \f]
		/// @param x new solution vector
		void update_quantities(const Eigen::VectorXd &x) override;

		/// @brief Compute the predicted solution to be used in the inertia term \f$(x-\tilde{x})^TM(x-\tilde{x})\f$.
		/// \f[
		/// 	\tilde{x} = x^t + \Delta t v^t
		/// \f]
		/// @return value for \f$\tilde{x}\f$
		Eigen::VectorXd x_tilde() const override;

		/// @brief Compute the current velocity given the current solution and using the stored previous solution(s).
		/// \f[
		/// 	v = \frac{x - x^t}{\Delta t}
		/// \f]
		/// @param x current solution vector
		/// @return value for \f$v\f$
		Eigen::VectorXd compute_velocity(const Eigen::VectorXd &x) const override;

		/// @brief Compute the current acceleration given the current velocity and using the stored previous velocity(s).
		/// \f[
		/// 	a = \frac{v - v^t}{\Delta t}
		/// \f]
		/// @param v current velocity
		/// @return value for \f$a\f$
		Eigen::VectorXd compute_acceleration(const Eigen::VectorXd &v) const override;

		/// @brief Compute the acceleration scaling used to scale forces when integrating a second order ODE.
		/// \f[
		/// 	\Delta t^2
		/// \f]
		double acceleration_scaling() const override;

		/// @brief Compute the derivative of the velocity with respect to the solution.
		/// \f[
		/// 	\frac{\partial v}{\partial x} = \frac{1}{\Delta t}
		/// \f]
		/// \f[
		/// 	\frac{\partial v}{\partial x^t} = \frac{-1}{\Delta t}
		/// \f]
		/// \f[
		/// 	\frac{\partial v}{\partial x^{t-1}} = 0
		/// \f]
		/// @param prev_ti index of the previous solution to use (0 -> current; 1 -> previous; 2 -> second previous; etc.)
		double dv_dx(const unsigned prev_ti = 0) const override;
	};
} // namespace polyfem::time_integrator

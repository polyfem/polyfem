#pragma once

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

namespace polyfem::time_integrator
{
	/// Implicit Euler time integrator of a second order ODE (equivently a system of coupled first order ODEs).
	/// \f[
	/// 	x^{t+1} = x^t + \Delta t v^{t+1}\\
	/// 	v^{t+1} = v^t + \Delta t a^{t+1}
	/// \f]
	/// @see https://en.wikipedia.org/wiki/Backward_Euler_method
	class ImplicitEuler : public ImplicitTimeIntegrator
	{
	public:
		ImplicitEuler() {}

		/// @brief Update the time integration quantaties (i.e., \f$x\f$, \f$v\f$, and \f$a\f$).
		/// \f[
		/// 	v^{t+1} = \frac{1}{\Delta t} (x - x^t)\\
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

		/// @brief Compute the acceleration scaling used to scale forces when integrating a second order ODE.
		/// \f[
		/// 	\Delta t^2
		/// \f]
		double acceleration_scaling() const override;
	};
} // namespace polyfem::time_integrator

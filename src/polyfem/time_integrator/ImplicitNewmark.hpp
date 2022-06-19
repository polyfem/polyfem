#pragma once

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

namespace polyfem::time_integrator
{
	/// Implicit Newmark-beta method.
	/// \f[
	/// 	x^{t+1} = x^t + \Delta t v^t + \frac{\Delta t^2}{2}((1-2\beta)a^t + 2 \beta a^{t+1})\\
	/// 	v^{t+1} = v^t + (1-\gamma)\Delta ta^t + \gamma \Delta ta^{t+1}
	/// \f]
	/// @see https://en.wikipedia.org/wiki/Newmark-beta_method
	class ImplicitNewmark : public ImplicitTimeIntegrator
	{
	public:
		ImplicitNewmark() {}

		/// @brief Set the `gamma` and `beta` parameters from a json object.
		/// @param params json containing `{"gamma": 0.5, "beta": 0.25}`
		void set_parameters(const nlohmann::json &params) override;

		/// @brief Update the time integration quantaties (i.e., \f$x\f$, \f$v\f$, and \f$a\f$).
		/// @param x new solution vector
		void update_quantities(const Eigen::VectorXd &x) override;

		/// @brief Compute the predicted solution to be used in the inertia term \f$(x-\tilde{x})^TM(x-\tilde{x})\f$.
		/// \f[
		/// 	\tilde{x} = x^t + \Delta t (v^t + (0.5 - \beta) \Delta t a^t)
		/// \f]
		/// @return value for \f$\tilde{x}\f$
		Eigen::VectorXd x_tilde() const override;

		/// @brief Compute the acceleration scaling used to scale forces when integrating a second order ODE.
		/// \f[
		/// 	\beta \Delta t^2
		/// \f]
		double acceleration_scaling() const override;

		/// @brief \f$\beta\f$ parameter for blending accelerations in the solution update.
		double beta() const { return m_beta; }
		/// @brief \f$\gamma\f$ parameter for blending accelerations in the velocity update.
		double gamma() const { return m_gamma; }

	protected:
		/// @brief \f$\beta\f$ parameter for blending accelerations in the solution update.
		double m_beta = 0.25;
		/// @brief \f$\gamma\f$ parameter for blending accelerations in the velocity update.
		double m_gamma = 0.5;
	};
} // namespace polyfem::time_integrator

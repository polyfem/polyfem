#pragma once

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

namespace polyfem::time_integrator
{
	/// @brief Backward Differential Formulas
	/// \f[
	/// 	x^{t+1} = \left(\sum_{i=0}^{n-1} \alpha_{i} x^{t-i}\right)+ \Delta t \beta v^{t+1}\\
	/// 	v^{t+1} = \left(\sum_{i=0}^{n-1} \alpha_{i} v^{t-i}\right)+ \Delta t \beta a^{t+1}
	/// \f]
	/// @see https://en.wikipedia.org/wiki/Backward_differentiation_formula
	class BDF : public ImplicitTimeIntegrator
	{
	public:
		BDF() {}

		/// @brief Set the number of steps parameters from a json object.
		/// @param params json containing `{"num_steps": 1}`
		void set_parameters(const nlohmann::json &params) override;

		using ImplicitTimeIntegrator::init;

		/// @brief Update the time integration quantaties (i.e., \f$x\f$, \f$v\f$, and \f$a\f$).
		/// @param x_prevs vector of previous solutions
		/// @param v_prevs vector of previous velocities
		/// @param a_prevs vector of previous accelerations
		/// @param dt time step
		void init(const std::vector<Eigen::VectorXd> &x_prevs,
				  const std::vector<Eigen::VectorXd> &v_prevs,
				  const std::vector<Eigen::VectorXd> &a_prevs,
				  double dt);

		/// @brief Update the time integration quantaties (i.e., \f$x\f$, \f$v\f$, and \f$a\f$).
		/// @param x new solution vector
		void update_quantities(const Eigen::VectorXd &x) override;

		/// @brief Compute the predicted solution to be used in the inertia term \f$(x-\tilde{x})^TM(x-\tilde{x})\f$.
		/// \f[
		/// 	\tilde{x} = \left(\sum_{i=0}^{n-1} \alpha_i x^{t-i}\right) + \beta \Delta t \left(\sum_{i=0}^{n-1} \alpha_i v^{t-i}\right)
		/// \f]
		/// @return value for \f$\tilde{x}\f$
		Eigen::VectorXd x_tilde() const override;

		/// @brief Compute the acceleration scaling used to scale forces when integrating a second order ODE.
		/// \f[
		/// 	\beta^2 \Delta t^2
		/// \f]
		double acceleration_scaling() const override;

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

	protected:
		int num_steps;

		/// @brief Retrieve the alphas used for BDF with `i` steps.
		/// @param i number of steps
		/// @see https://en.wikipedia.org/wiki/Backward_differentiation_formula#General_formula
		static const std::vector<double> &alphas(const int i);

		/// @brief Retrieve the value of beta used for BDF with `i` steps.
		/// @param i number of steps
		/// @see https://en.wikipedia.org/wiki/Backward_differentiation_formula#General_formula
		static double betas(const int i);
	};
} // namespace polyfem::time_integrator

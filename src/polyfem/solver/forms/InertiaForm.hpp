#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#include <Eigen/Core>

#include <functional>

namespace polyfem::solver
{
	/// @brief Form of the inertia
	class InertiaForm : public Form
	{
		friend class InertiaForceDerivative;

	public:
		/// @brief Construct a new Inertia Form object
		/// @param mass Mass matrix
		/// @param time_integrator Time integrator
		InertiaForm(const StiffnessMatrix &mass,
					const time_integrator::ImplicitTimeIntegrator &time_integrator);

		std::string name() const override { return "inertia"; }

		using XTildeUpdater = std::function<void(const double, const Eigen::VectorXd &, Eigen::VectorXd &)>;
		// Optional hook for fields whose time-dependent essential boundary values require a
		// lifted time-integrator prediction before applying the mass term.
		void set_x_tilde_updater(XTildeUpdater updater);

		void update_quantities(const double t, const Eigen::VectorXd &x) override;

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
		Eigen::VectorXd x_tilde() const;

		// TODO mass might be time dependent
		const StiffnessMatrix &mass_;                                    ///< Mass matrix
		const time_integrator::ImplicitTimeIntegrator &time_integrator_; ///< Time integrator
		XTildeUpdater x_tilde_updater_;
		Eigen::VectorXd x_tilde_;
	};
} // namespace polyfem::solver

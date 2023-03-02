#include "InertiaForm.hpp"

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

namespace polyfem::solver
{
	InertiaForm::InertiaForm(const StiffnessMatrix &mass,
							 const time_integrator::ImplicitTimeIntegrator &time_integrator)
		: mass_(mass), time_integrator_(time_integrator)
	{
		assert(mass.size() != 0);
	}

	double InertiaForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd tmp = x - time_integrator_.x_tilde();
		// FIXME: DBC on x tilde
		const double prod = tmp.transpose() * mass_ * tmp;
		const double energy = 0.5 * prod;
		return energy;
	}

	void InertiaForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = mass_ * (x - time_integrator_.x_tilde());
	}

	void InertiaForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = mass_;
	}
} // namespace polyfem::solver

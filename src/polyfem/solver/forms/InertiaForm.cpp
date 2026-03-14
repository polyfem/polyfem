#include "InertiaForm.hpp"

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/Types.hpp>

#include <cassert>
#include <utility>

namespace polyfem::solver
{
	InertiaForm::InertiaForm(const StiffnessMatrix &mass,
							 const time_integrator::ImplicitTimeIntegrator &time_integrator)
		: mass_(mass), time_integrator_(time_integrator), x_tilde_(time_integrator.x_tilde())
	{
		assert(mass.size() != 0);
		assert(x_tilde_.size() == mass.rows());
	}

	void InertiaForm::set_x_tilde_updater(XTildeUpdater updater)
	{
		x_tilde_updater_ = std::move(updater);
	}

	Eigen::VectorXd InertiaForm::x_tilde() const
	{
		assert(x_tilde_.size() == mass_.rows());
		return x_tilde_;
	}

	void InertiaForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		x_tilde_ = time_integrator_.x_tilde();

		if (x_tilde_updater_)
			x_tilde_updater_(t, x, x_tilde_);

		assert(x_tilde_.size() == mass_.rows());
	}

	double InertiaForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd tmp = x - x_tilde();
		const double prod = tmp.transpose() * mass_ * tmp;
		const double energy = 0.5 * prod;
		return energy;
	}

	void InertiaForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = mass_ * (x - x_tilde());
	}

	void InertiaForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = mass_;
	}

	void InertiaForm::accumulateHessian(const double weight, const Eigen::VectorXd &x, NewtonHessian &H, SystemAssembler<2> &assembler) const
	{
		if (m_useLumpedMass) H.H_ss->addDiag((weight * M_lumped).eval());
            else             H.H_ss->addWithSubSparsityFast(*(M_full.H_ss), weight);
	}

	void InertiaForm::accumulateHessian(const double weight, const Eigen::VectorXd &x, NewtonHessian &H, SystemAssembler<3> &assembler) const
	{
		if (m_useLumpedMass) H.H_ss->addDiag((weight * M_lumped).eval());
            else             H.H_ss->addWithSubSparsityFast(*(M_full.H_ss), weight);
	}

	NewtonHessian InertiaForm::hessianSparsityPattern(SystemAssembler<2> &assembler) const
	{
		return M_full;
	}

	NewtonHessian InertiaForm::hessianSparsityPattern(SystemAssembler<3> &assembler) const
	{
		return M_full;
	}

	
} // namespace polyfem::solver

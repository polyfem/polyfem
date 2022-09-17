#include "DampingForm.hpp"

#include <polyfem/utils/Timer.hpp>

namespace polyfem::solver
{

	DampingForm::DampingForm(const State &state, const double dt)
		: state_(state), assembler_(state.damping_assembler), dt_(dt)
	{
	}

	double DampingForm::value_unweighted(const Eigen::VectorXd &x) const
	{
        if (x_prev.size() != x.size())
            return 0;
		return assembler_.assemble(state_.mesh->is_volume(), dt_, state_.bases, state_.geom_bases(), state_.ass_vals_cache, x, x_prev);
	}

	void DampingForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
        if (x_prev.size() != x.size())
        {
            gradv.setZero(x.size());
        }
        else
        {
            Eigen::MatrixXd grad;
            assembler_.assemble_grad(state_.mesh->is_volume(), state_.n_bases, dt_, state_.bases, state_.geom_bases(), state_.ass_vals_cache, x, x_prev, grad);
            gradv = grad;
        }
	}

	void DampingForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		POLYFEM_SCOPED_TIMER("\telastic hessian");

		hessian.resize(x.size(), x.size());

        if (x_prev.size() == x.size())
		    assembler_.assemble_hessian(state_.mesh->is_volume(), state_.n_bases, dt_, false, state_.bases, state_.geom_bases(), state_.ass_vals_cache, x, x_prev, mat_cache_, hessian);
	}

	void DampingForm::init_lagging(const Eigen::VectorXd &x)
	{
		x_prev = x;
	}
} // namespace polyfem::solver

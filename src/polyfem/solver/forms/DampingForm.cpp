#include "DampingForm.hpp"

#include <polyfem/utils/Timer.hpp>

namespace polyfem::solver
{

	DampingForm::DampingForm(const State &state, const double dt)
		: state_(state), assembler_(state.assembler), dt_(dt)
	{
	}

	double DampingForm::value_unweighted(const Eigen::VectorXd &x) const
	{
        if (x_prev.size() != x.size())
            return 0;
		return assembler_.assemble_transient_energy("Damping", state_.mesh->is_volume(), dt_, state_.bases, state_.geom_bases(), state_.ass_vals_cache, x, x_prev);
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
            assembler_.assemble_transient_energy_gradient("Damping", state_.mesh->is_volume(), dt_, state_.n_bases, state_.bases, state_.geom_bases(), state_.ass_vals_cache, x, x_prev, grad);
            gradv = grad;
        }
	}

	void DampingForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		POLYFEM_SCOPED_TIMER("\tdamping hessian");

		hessian.resize(x.size(), x.size());

        if (x_prev.size() == x.size())
		    assembler_.assemble_transient_energy_hessian("Damping", state_.mesh->is_volume(), dt_, state_.n_bases, false, state_.bases, state_.geom_bases(), state_.ass_vals_cache, x, x_prev, mat_cache_, hessian);
	}
} // namespace polyfem::solver

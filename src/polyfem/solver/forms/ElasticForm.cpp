#include "ElasticForm.hpp"

#include <polyfem/utils/Timer.hpp>

namespace polyfem::solver
{

	ElasticForm::ElasticForm(const State &state)
		: state_(state), assembler_(state.assembler), formulation_(state_.formulation())
	{
	}

	double ElasticForm::value(const Eigen::VectorXd &x)
	{
		return assembler_.assemble_energy(
			formulation_, state_.mesh->is_volume(), state_.bases, state_.geom_bases(),
			state_.ass_vals_cache, x);
	}

	void ElasticForm::first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
	{
		Eigen::MatrixXd grad;
		assembler_.assemble_energy_gradient(
			formulation_, state_.mesh->is_volume(), state_.n_bases, state_.bases, state_.geom_bases(),
			state_.ass_vals_cache, x, grad);
		gradv = grad;
	}

	void ElasticForm::second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		POLYFEM_SCOPED_TIMER("\telastic hessian");

		hessian.resize(x.size(), x.size());

		if (assembler_.is_linear(formulation_))
		{
			compute_cached_stiffness();
			hessian = cached_stiffness_;
		}
		else
		{
			assembler_.assemble_energy_hessian(
				formulation_, state_.mesh->is_volume(), state_.n_bases, project_to_psd_, state_.bases,
				state_.geom_bases(), state_.ass_vals_cache, x, mat_cache_, hessian);
		}
	}

	bool ElasticForm::is_step_valid(const Eigen::VectorXd &, const Eigen::VectorXd &x1)
	{
		Eigen::VectorXd grad;
		first_derivative(x1, grad);

		if (grad.array().isNaN().any())
			return false;

		// Check the scalar field in the output does not contain NANs.
		// WARNING: Does not work because the energy is not evaluated at the same quadrature points.
		//          This causes small step lengths in the LS.
		// TVector x1_full;
		// reduced_to_full(x1, x1_full);
		// return state_.check_scalar_value(x1_full, true, false);
		return true;
	}

	void ElasticForm::compute_cached_stiffness()
	{
		if (cached_stiffness_.size() == 0)
		{
			if (assembler_.is_linear(state_.formulation()))
			{
				assembler_.assemble_problem(
					formulation_, state_.mesh->is_volume(), state_.n_bases, state_.bases, state_.geom_bases(),
					state_.ass_vals_cache, cached_stiffness_);
			}
		}
	}
} // namespace polyfem::solver

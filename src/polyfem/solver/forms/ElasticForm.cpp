#include "ElasticForm.hpp"

#include <polyfem/utils/Timer.hpp>

namespace polyfem::solver
{

	ElasticForm::ElasticForm(const State &state)
		: state_(state), assembler_(state.assembler), formulation_(state_.formulation())
	{
		if (assembler_.is_linear(formulation()))
			compute_cached_stiffness();
	}

	double ElasticForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return assembler_.assemble_energy(
			formulation(), state_.mesh->is_volume(), state_.bases, state_.geom_bases(),
			state_.ass_vals_cache, x);
	}

	void ElasticForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		Eigen::MatrixXd grad;
		assembler_.assemble_energy_gradient(
			formulation(), state_.mesh->is_volume(), state_.n_bases, state_.bases, state_.geom_bases(),
			state_.ass_vals_cache, x, grad);
		gradv = grad;
	}

	void ElasticForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		POLYFEM_SCOPED_TIMER("\telastic hessian");

		hessian.resize(x.size(), x.size());

		if (assembler_.is_linear(formulation()))
		{
			assert(cached_stiffness_.rows() != x.size() && cached_stiffness_.cols() != x.size());
			hessian = cached_stiffness_;
		}
		else
		{
			// TODO: somehow remove mat_cache_ so this function can be marked const
			assembler_.assemble_energy_hessian(
				formulation(), state_.mesh->is_volume(), state_.n_bases, project_to_psd_, state_.bases,
				state_.geom_bases(), state_.ass_vals_cache, x, mat_cache_, hessian);
		}
	}

	bool ElasticForm::is_step_valid(const Eigen::VectorXd &, const Eigen::VectorXd &x1) const
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
		if (assembler_.is_linear(formulation()) && cached_stiffness_.size() == 0)
		{
			assembler_.assemble_problem(
				formulation(), state_.mesh->is_volume(), state_.n_bases, state_.bases, state_.geom_bases(),
				state_.ass_vals_cache, cached_stiffness_);
		}
	}
} // namespace polyfem::solver

#include "ElasticForm.hpp"

#include <polyfem/utils/Timer.hpp>

namespace polyfem
{
	namespace solver
	{

		ElasticForm::ElasticForm(const State &state)
			: state_(state), assembler_(state.assembler), formulation_(state_.formulation())
		{
		}

		double ElasticForm::value(const Eigen::VectorXd &x)
		{
			const auto &gbases = state_.iso_parametric() ? state_.bases : state_.geom_bases;
			const double energy = assembler_.assemble_energy(formulation_, state_.mesh->is_volume(), state_.bases, gbases, state_.ass_vals_cache, x);

			return energy;
		}

		void ElasticForm::first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
		{
			Eigen::MatrixXd grad;
			const auto &gbases = state_.iso_parametric() ? state_.bases : state_.geom_bases;
			assembler_.assemble_energy_gradient(formulation_, state_.mesh->is_volume(), state_.n_bases, state_.bases, gbases, state_.ass_vals_cache, x, grad);
			gradv = grad;
		}

		void ElasticForm::second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
		{
			const auto full_size = x.size();
			hessian.resize(full_size, full_size);
			POLYFEM_SCOPED_TIMER("\telastic hessian time");

			const auto &gbases = state_.iso_parametric() ? state_.bases : state_.geom_bases;
			if (assembler_.is_linear(formulation_))
			{
				compute_cached_stiffness();
				hessian = cached_stiffness_;
			}
			else
			{
				assembler_.assemble_energy_hessian(formulation_, state_.mesh->is_volume(), state_.n_bases, project_to_psd_, state_.bases, gbases, state_.ass_vals_cache, x, mat_cache_, hessian);
			}
		}

		bool ElasticForm::is_step_valid(const Eigen::VectorXd &, const Eigen::VectorXd &x1)
		{
			Eigen::VectorXd grad(x1.size());
			first_derivative(x1, grad);

			if (std::isnan(grad.norm()))
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
				const auto &gbases = state_.iso_parametric() ? state_.bases : state_.geom_bases;
				if (assembler_.is_linear(state_.formulation()))
				{
					assembler_.assemble_problem(formulation_, state_.mesh->is_volume(), state_.n_bases, state_.bases, gbases, state_.ass_vals_cache, cached_stiffness_);
				}
			}
		}
	} // namespace solver
} // namespace polyfem

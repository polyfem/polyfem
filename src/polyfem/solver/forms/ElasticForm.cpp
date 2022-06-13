#pragma once

#include <polyfem/utils/Types.hpp>

namespace polyfem
{
	namespace solver
	{

		ElasticForm::ElasticForm()
		{
		}

		double ElasticForm::value(const Eigen::VectorXd &x)
		{
			const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
			const double energy = assembler.assemble_energy(rhs_assembler.formulation(), state.mesh->is_volume(), state.bases, gbases, state.ass_vals_cache, full);

			return energy;
		}

		void ElasticForm::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
		{
			const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
			assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, state.ass_vals_cache, full, grad);
		}

		virtual void hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
		{
			const auto full_size = x.size();
			hessian.rezie(full_size, full_size);
			POLYFEM_SCOPED_TIMER("\telastic hessian time");

			const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
			if (assembler.is_linear(rhs_assembler.formulation()))
			{
				compute_cached_stiffness();
				hessian = cached_stiffness;
			}
			else
			{
				assembler.assemble_energy_hessian(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, project_to_psd, state.bases, gbases, state.ass_vals_cache, full, mat_cache, hessian);
			}
		}

		bool NLProblem::is_step_valid(const TVector &x0, const TVector &x1)
		{
			TVector grad = TVector::Zero(reduced_size);
			gradient(x1, grad, true);

			if (std::isnan(grad.norm()))
				return false;

			// Check the scalar field in the output does not contain NANs.
			// WARNING: Does not work because the energy is not evaluated at the same quadrature points.
			//          This causes small step lengths in the LS.
			// TVector x1_full;
			// reduced_to_full(x1, x1_full);
			// return state.check_scalar_value(x1_full, true, false);
			return true;
		}

		void NLProblem::compute_cached_stiffness()
		{
			if (cached_stiffness.size() == 0)
			{
				const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
				if (assembler.is_linear(state.formulation()))
				{
					assembler.assemble_problem(state.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, state.ass_vals_cache, cached_stiffness);
				}
			}
		}
	} // namespace solver
} // namespace polyfem

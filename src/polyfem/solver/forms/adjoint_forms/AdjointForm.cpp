#include "AdjointForm.hpp"
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/State.hpp>
#include <polyfem/assembler/Assembler.hpp>

#include <polyfem/solver/forms/parametrization/SDFParametrizations.hpp>

namespace polyfem::solver
{
	double AdjointForm::value(const Eigen::VectorXd &x) const
	{
		double val = Form::value(x);
		if (print_energy_ == 1)
		{
			logger().debug("[{}] {}", print_energy_keyword_, val);
			print_energy_ = 2;
		}
		return val;
	}

	void AdjointForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		if (print_energy_ == 2)
			print_energy_ = 1;
	}

	void AdjointForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		log_and_throw_error("Not implemented");
	}

	Eigen::MatrixXd AdjointForm::compute_reduced_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const
	{
		Eigen::MatrixXd rhs = compute_adjoint_rhs_unweighted(x, state);
		if (!state.problem->is_time_dependent() && !state.lin_solver_cached) // nonlinear static solve only
		{
			Eigen::MatrixXd reduced;
			for (int i = 0; i < rhs.cols(); i++)
			{
				Eigen::VectorXd reduced_vec = state.solve_data.nl_problem->full_to_reduced(rhs.col(i));
				if (i == 0)
					reduced.setZero(reduced_vec.rows(), rhs.cols());
				reduced.col(i) = reduced_vec;
			}
			return reduced;
		}
		else
			return rhs;
	}

	void AdjointForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv.setZero(x.size());
		for (const auto &param_map : variable_to_simulations_)
		{
			auto adjoint_term = param_map->compute_adjoint_term(x);
			gradv += adjoint_term;
		}

		gradv /= weight();

		Eigen::VectorXd partial_grad;
		compute_partial_gradient_unweighted(x, partial_grad);
		gradv += partial_grad;
	}

	void AdjointForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = Eigen::VectorXd::Zero(x.size());
	}

	Eigen::MatrixXd AdjointForm::compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const
	{
		return Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
	}

	double StaticForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return value_unweighted_step(0, x);
	}

	void StaticForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		compute_partial_gradient_unweighted_step(0, x, gradv);
	}

	Eigen::VectorXd StaticForm::compute_adjoint_rhs_unweighted_step_prev(const int time_step, const Eigen::VectorXd &x, const State &state) const
	{
		return Eigen::MatrixXd::Zero(state.ndof(), 1);
	}

	Eigen::MatrixXd StaticForm::compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const
	{
		assert(!depends_on_step_prev());
		Eigen::MatrixXd term = Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
		term.col(0) = compute_adjoint_rhs_unweighted_step(0, x, state);

		return term;
	}

	double MaxStressForm::value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd max_stress;
		max_stress.setZero(state_.bases.size());
		utils::maybe_parallel_for(state_.bases.size(), [&](int start, int end, int thread_id) {
			Eigen::MatrixXd local_vals;
			assembler::ElementAssemblyValues vals;
			for (int e = start; e < end; e++)
			{
				if (interested_ids_.size() != 0 && interested_ids_.find(state_.mesh->get_body_id(e)) == interested_ids_.end())
					continue;

				state_.ass_vals_cache.compute(e, state_.mesh->is_volume(), state_.bases[e], state_.geom_bases()[e], vals);
				// std::vector<assembler::Assembler::NamedMatrix> result;
				// state_.assembler->compute_tensor_value(e, state_.bases[e], state_.geom_bases()[e], vals.quadrature.points, state_.diff_cached.u(time_step), result);
				std::dynamic_pointer_cast<assembler::ElasticityAssembler>(state_.assembler)->compute_stress_tensor(e, state_.bases[e], state_.geom_bases()[e], vals.quadrature.points, state_.diff_cached.u(time_step), ElasticityTensorType::PK1, local_vals);

				Eigen::VectorXd stress_norms = local_vals.rowwise().norm();
				max_stress(e) = std::max(max_stress(e), stress_norms.maxCoeff());
			}
		});

		return max_stress.maxCoeff();
	}
	Eigen::VectorXd MaxStressForm::compute_adjoint_rhs_unweighted_step(const int time_step, const Eigen::VectorXd &x, const State &state) const
	{
		log_and_throw_error("MaxStressForm is not differentiable!");
		return Eigen::VectorXd();
	}
	void MaxStressForm::compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		log_and_throw_error("MaxStressForm is not differentiable!");
	}
} // namespace polyfem::solver
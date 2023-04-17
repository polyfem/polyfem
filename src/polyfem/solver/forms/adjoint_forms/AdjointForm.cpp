#include "AdjointForm.hpp"
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>
#include <polyfem/State.hpp>

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
				Eigen::VectorXd reduced_vec = state.solve_data.nl_problem->full_to_reduced_grad(rhs.col(i));
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

		gradv /= weight_;

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

	Eigen::MatrixXd StaticForm::compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const
	{
		Eigen::MatrixXd term = Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
		term.col(time_step_) = compute_adjoint_rhs_unweighted_step(x, state);

		return term;
	}

	NodeTargetForm::NodeTargetForm(const State &state, const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const json &args) : StaticForm(variable_to_simulations), state_(state)
	{
		std::string target_data_path = args["target_data_path"];
		if (!std::filesystem::is_regular_file(target_data_path))
		{
			throw std::runtime_error("Marker path invalid!");
		}
		Eigen::MatrixXd tmp;
		io::read_matrix(target_data_path, tmp);

		// markers to nodes
		Eigen::VectorXi nodes = tmp.col(0).cast<int>();
		target_vertex_positions.setZero(nodes.size(), state_.mesh->dimension());
		active_nodes.reserve(nodes.size());
		for (int s = 0; s < nodes.size(); s++)
		{
			const int node_id = state_.in_node_to_node(nodes(s));
			target_vertex_positions.row(s) = tmp.block(s, 1, 1, tmp.cols() - 1);
			active_nodes.push_back(node_id);
		}
	}
	NodeTargetForm::NodeTargetForm(const State &state, const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const std::vector<int> &active_nodes_, const Eigen::MatrixXd &target_vertex_positions_) : StaticForm(variable_to_simulations), state_(state), target_vertex_positions(target_vertex_positions_), active_nodes(active_nodes_)
	{
	}
	Eigen::VectorXd NodeTargetForm::compute_adjoint_rhs_unweighted_step(const Eigen::VectorXd &x, const State &state) const
	{
		Eigen::VectorXd rhs;
		rhs.setZero(state.diff_cached.u(0).size());

		const int dim = state_.mesh->dimension();

		if (&state == &state_)
		{
			int i = 0;
			Eigen::VectorXd disp = state_.diff_cached.u(time_step_);
			for (int v : active_nodes)
			{
				RowVectorNd cur_pos = state_.mesh_nodes->node_position(v) + disp.segment(v * dim, dim).transpose();

				rhs.segment(v * dim, dim) = 2 * (cur_pos - target_vertex_positions.row(i++));
			}
		}

		return rhs;
	}
	double NodeTargetForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const int dim = state_.mesh->dimension();
		double val = 0;
		int i = 0;
		Eigen::VectorXd disp = state_.diff_cached.u(time_step_);
		for (int v : active_nodes)
		{
			RowVectorNd cur_pos = state_.mesh_nodes->node_position(v) + disp.segment(v * dim, dim).transpose();
			val += (cur_pos - target_vertex_positions.row(i++)).squaredNorm();
		}
		return val;
	}
	void NodeTargetForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv.setZero(x.size());
		for (const auto &param_map : variable_to_simulations_)
		{
			const auto &param_type = param_map->get_parameter_type();

			for (const auto &state : param_map->get_states())
			{
				if (state.get() != &state_)
					continue;

				Eigen::VectorXd term;
				if (param_type == ParameterType::Shape)
					throw std::runtime_error("Shape derivative of NodeTargetForm not implemented!");

				if (term.size() > 0)
					gradv += param_map->apply_parametrization_jacobian(term, x);
			}
		}
	}

	double MaxStressForm::value_unweighted(const Eigen::VectorXd &x) const
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
				state_.assembler.compute_tensor_value(state_.formulation(), e, state_.bases[e], state_.geom_bases()[e], vals.quadrature.points, state_.diff_cached.u(time_step_), local_vals);
				Eigen::VectorXd stress_norms = local_vals.rowwise().norm();
				max_stress(e) = std::max(max_stress(e), stress_norms.maxCoeff());
			}
		});

		return max_stress.maxCoeff();
	}
	Eigen::VectorXd MaxStressForm::compute_adjoint_rhs_unweighted_step(const Eigen::VectorXd &x, const State &state) const
	{
		log_and_throw_error("MaxStressForm is not differentiable!");
		return Eigen::VectorXd();
	}
	
	double HomogenizedDispGradForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return state_.diff_cached.disp_grad()(dimensions_[0], dimensions_[1]);
	}
	Eigen::MatrixXd HomogenizedDispGradForm::compute_reduced_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const
	{
		if (&state != &state_)
		{
			if (!state.problem->is_time_dependent() && !state.lin_solver_cached) // nonlinear static solve only
			{
				return state.solve_data.nl_problem->full_to_reduced_grad(Eigen::VectorXd::Zero(state.ndof()));
			}
			else
			{
				return Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size());
			}
		}

		std::shared_ptr<NLHomoProblem> problem = std::dynamic_pointer_cast<NLHomoProblem>(state_.solve_data.nl_problem);
		if (!problem)
			log_and_throw_error("Homogenized displacement gradient objective only works in homogenization!");
		
		Eigen::MatrixXd disp_grad;
		disp_grad.setZero(state.mesh->dimension(), state.mesh->dimension());
		disp_grad(dimensions_[0], dimensions_[1]) = 1;
		
		Eigen::VectorXd rhs;
		rhs.setZero(problem->reduced_size() + problem->macro_reduced_size());
		rhs.tail(problem->macro_reduced_size()) = problem->macro_full_to_reduced_grad(utils::flatten(disp_grad));

		return rhs;
	}

	WeightedSolution::WeightedSolution(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : AdjointForm(variable_to_simulations), state_(state)
	{
		coeffs.setRandom(state_.ndof());
	}
	double WeightedSolution::value_unweighted(const Eigen::VectorXd &x) const
	{
		return state_.diff_cached.u(0).dot(coeffs);
	}
	Eigen::MatrixXd WeightedSolution::compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const
	{
		return coeffs;
	}
	void WeightedSolution::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv.setZero(x.size());
		for (const auto &param_map : variable_to_simulations_)
		{
			const auto &param_type = param_map->get_parameter_type();

			for (const auto &state : param_map->get_states())
			{
				if (state.get() != &state_)
					continue;

				Eigen::VectorXd term;
				if (param_type == ParameterType::PeriodicShape)
				{
					auto adjoint_rhs = compute_adjoint_rhs_unweighted(x, state_);
					std::shared_ptr<NLHomoProblem> homo_problem = std::dynamic_pointer_cast<NLHomoProblem>(state_.solve_data.nl_problem);
					term = homo_problem->reduced_to_full_shape_derivative(state_.diff_cached.disp_grad(), adjoint_rhs);
					term = utils::flatten(utils::unflatten(term, state_.mesh->dimension())(state_.primitive_to_node(), Eigen::all));
					term = state_.periodic_mesh_map->apply_jacobian(term, state_.periodic_mesh_representation);
				}
				if (term.size() > 0)
					gradv += param_map->apply_parametrization_jacobian(term, x);
			}
		}
	}
} // namespace polyfem::solver
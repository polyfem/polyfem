#include "VariableToSimulation.hpp"
#include <polyfem/State.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/solver/Optimizations.hpp>

#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

namespace polyfem::solver
{
	std::unique_ptr<VariableToSimulation> VariableToSimulation::create(const std::string &type, const std::vector<std::shared_ptr<State>> &states, CompositeParametrization &&parametrization)
	{
		if (type == "shape")
			return std::make_unique<ShapeVariableToSimulation>(states, parametrization);
		else if (type == "elastic")
			return std::make_unique<ElasticVariableToSimulation>(states, parametrization);
		else if (type == "friction")
			return std::make_unique<FrictionCoeffientVariableToSimulation>(states, parametrization);
		else if (type == "damping")
			return std::make_unique<DampingCoeffientVariableToSimulation>(states, parametrization);
		else if (type == "initial")
			return std::make_unique<InitialConditionVariableToSimulation>(states, parametrization);
		else if (type == "dirichlet")
			return std::make_unique<DirichletVariableToSimulation>(states, parametrization);
		else if (type == "pressure")
			return std::make_unique<PressureVariableToSimulation>(states, parametrization);

		log_and_throw_adjoint_error("Invalid type of VariableToSimulation!");
		return std::unique_ptr<VariableToSimulation>();
	}

	Eigen::VectorXi VariableToSimulation::get_output_indexing(const Eigen::VectorXd &x) const
	{
		const int out_size = parametrization_.size(x.size());
		if (output_indexing_.size() == out_size)
			return output_indexing_;
		else if (output_indexing_.size() == 0)
		{
			Eigen::VectorXi ind;
			ind.setLinSpaced(out_size, 0, out_size - 1);
			return ind;
		}
		else
			log_and_throw_adjoint_error(fmt::format("[{}] Indexing size and output size of the Parametrization do not match! {} vs {}", name(), output_indexing_.size(), out_size));
		return Eigen::VectorXi();
	}

	Eigen::VectorXd VariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const
	{
		return parametrization_.apply_jacobian(term(get_output_indexing(x)), x);
	}

	Eigen::VectorXd VariableToSimulation::inverse_eval()
	{
		log_and_throw_adjoint_error("[{}] inverse_eval not implemented!", name());
		return Eigen::VectorXd();
	}

	void VariableToSimulationGroup::init(const json &args, const std::vector<std::shared_ptr<State>> &states, const std::vector<int> &variable_sizes)
	{
		std::vector<ValueType>().swap(L);
		for (const auto &arg : args)
			L.push_back(AdjointOptUtils::create_variable_to_simulation(arg, states, variable_sizes));
	}

	Eigen::VectorXd VariableToSimulationGroup::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd adjoint_term = Eigen::VectorXd::Zero(x.size());
		for (const auto &v2s : L)
			adjoint_term += v2s->compute_adjoint_term(x);
		return adjoint_term;
	}

	void VariableToSimulationGroup::compute_state_variable(const ParameterType type, const State *state_ptr, const Eigen::VectorXd &x, Eigen::VectorXd &state_variable) const
	{
		for (const auto &v2s : L)
		{
			if (v2s->get_parameter_type() != type)
				continue;

			const Eigen::VectorXd var = v2s->get_parametrization().eval(x);
			for (const auto &state : v2s->get_states())
			{
				if (state.get() != state_ptr)
					continue;

				state_variable(v2s->get_output_indexing(x)) = var;
			}
		}
	}

	Eigen::VectorXd VariableToSimulationGroup::apply_parametrization_jacobian(const ParameterType type, const State *state_ptr, const Eigen::VectorXd &x, const std::function<Eigen::VectorXd()> &grad) const
	{
		Eigen::VectorXd gradv = Eigen::VectorXd::Zero(x.size());
		Eigen::VectorXd raw_grad;
		for (const auto &v2s : L)
		{
			if (v2s->get_parameter_type() != type)
				continue;

			for (const auto &state : v2s->get_states())
			{
				if (state.get() != state_ptr)
					continue;

				if (raw_grad.size() == 0)
					raw_grad = v2s->apply_parametrization_jacobian(grad(), x);

				gradv += raw_grad;
			}
		}
		return gradv;
	}

	void ShapeVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		for (auto state : states_)
		{
			const int dim = state->mesh->dimension();

			// If indices include one vertex entry, we assume it include all entries of this vertex.
			for (int i = 0; i < indices.size(); i += dim)
				for (int j = 0; j < dim; j++)
					assert(indices(i + j) == indices(i) + j);

			for (int i = 0; i < indices.size(); i += dim)
				state->set_mesh_vertex(indices(i) / dim, state_variable(Eigen::seqN(i, dim)));
		}
	}
	Eigen::VectorXd ShapeVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
				AdjointTools::dJ_shape_transient_adjoint_term(*state, state->get_adjoint_mat(1), state->get_adjoint_mat(0), cur_term);
			else
				AdjointTools::dJ_shape_static_adjoint_term(*state, state->diff_cached.u(0), state->get_adjoint_mat(0), cur_term);

			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return apply_parametrization_jacobian(term, x);
	}
	Eigen::VectorXd ShapeVariableToSimulation::inverse_eval()
	{
		const int dim = states_[0]->mesh->dimension();
		const int npts = states_[0]->mesh->n_vertices();

		Eigen::VectorXd x;
		Eigen::VectorXi indices = get_output_indexing(x);

		if (indices.size() == 0)
			indices.setLinSpaced(npts * dim, 0, npts * dim - 1);

		Eigen::MatrixXd V, V_flat;
		states_[0]->get_vertices(V);
		V_flat = utils::flatten(V);

		x.setZero(indices.size());
		for (int i = 0; i < indices.size(); i++)
			x(i) = V_flat(indices(i));

		return parametrization_.inverse_eval(x);
	}

	void ElasticVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		for (auto state : states_)
		{
			const int n_elem = state->bases.size();
			assert(n_elem * 2 == state_variable.size());
			state->assembler->update_lame_params(state_variable.segment(0, n_elem), state_variable.segment(n_elem, n_elem));
		}
	}
	Eigen::VectorXd ElasticVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
				AdjointTools::dJ_material_transient_adjoint_term(*state, state->get_adjoint_mat(1), state->get_adjoint_mat(0), cur_term);
			else
				AdjointTools::dJ_material_static_adjoint_term(*state, state->diff_cached.u(0), state->get_adjoint_mat(0), cur_term);

			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return apply_parametrization_jacobian(term, x);
	}
	Eigen::VectorXd ElasticVariableToSimulation::inverse_eval()
	{
		auto &state = *(states_[0]);
		auto params_map = state.assembler->parameters();

		auto search_lambda = params_map.find("lambda");
		auto search_mu = params_map.find("mu");
		if (search_lambda == params_map.end() || search_mu == params_map.end())
		{
			log_and_throw_adjoint_error("[{}] Failed to find Lame parameters!", name());
			return Eigen::VectorXd();
		}

		Eigen::VectorXd lambdas(state.mesh->n_elements());
		Eigen::VectorXd mus(state.mesh->n_elements());
		for (int e = 0; e < state.mesh->n_elements(); e++)
		{
			RowVectorNd barycenter;
			if (!state.mesh->is_volume())
			{
				const auto &mesh2d = *dynamic_cast<mesh::Mesh2D *>(state.mesh.get());
				barycenter = mesh2d.face_barycenter(e);
			}
			else
			{
				const auto &mesh3d = *dynamic_cast<mesh::Mesh3D *>(state.mesh.get());
				barycenter = mesh3d.cell_barycenter(e);
			}
			lambdas(e) = search_lambda->second(RowVectorNd::Zero(state.mesh->dimension()), barycenter, 0., e);
			mus(e) = search_mu->second(RowVectorNd::Zero(state.mesh->dimension()), barycenter, 0., e);
		}
		state.assembler->update_lame_params(lambdas, mus);

		Eigen::VectorXd params(lambdas.size() + mus.size());
		params << lambdas, mus;

		return parametrization_.inverse_eval(params);
	}

	void FrictionCoeffientVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		assert(state_variable.size() == 1);
		assert(state_variable(0) >= 0);
		for (auto state : states_)
			state->args["contact"]["friction_coefficient"] = state_variable(0);
	}
	Eigen::VectorXd FrictionCoeffientVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
				AdjointTools::dJ_friction_transient_adjoint_term(*state, state->get_adjoint_mat(1), state->get_adjoint_mat(0), cur_term);
			else
				log_and_throw_adjoint_error("[{}] Gradient in static simulations not implemented!", name());

			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return apply_parametrization_jacobian(term, x);
	}
	Eigen::VectorXd FrictionCoeffientVariableToSimulation::inverse_eval()
	{
		log_and_throw_adjoint_error("[{}] inverse_eval not implemented!", name());
		return Eigen::VectorXd();
	}

	void DampingCoeffientVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		assert(state_variable.size() == 2);
		json damping_param = {
			{"psi", state_variable(0)},
			{"phi", state_variable(1)},
		};
		for (auto state : states_)
		{
			if (!state->args["materials"].is_array())
			{
				state->args["materials"]["psi"] = damping_param["psi"];
				state->args["materials"]["phi"] = damping_param["phi"];
			}
			else
			{
				for (auto &arg : state->args["materials"])
				{
					arg["psi"] = damping_param["psi"];
					arg["phi"] = damping_param["phi"];
				}
			}

			if (state->damping_assembler)
				state->damping_assembler->add_multimaterial(0, damping_param, state->units);
		}
		logger().info("[{}] Current params: {}, {}", name(), state_variable(0), state_variable(1));
	}
	Eigen::VectorXd DampingCoeffientVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
				AdjointTools::dJ_damping_transient_adjoint_term(*state, state->get_adjoint_mat(1), state->get_adjoint_mat(0), cur_term);
			else
				log_and_throw_adjoint_error("[{}] Static simulation not supported!", name());

			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return apply_parametrization_jacobian(term, x);
	}
	Eigen::VectorXd DampingCoeffientVariableToSimulation::inverse_eval()
	{
		log_and_throw_adjoint_error("[{}] inverse_eval not implemented!", name());
		return Eigen::VectorXd();
	}

	void InitialConditionVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		for (auto state : states_)
		{
			if (state_variable.size() != state->ndof() * 2)
			{
				log_and_throw_adjoint_error("[{}] Inconsistent number of parameters {} and number of dofs in forward {}!", name(), state_variable.size(), state->ndof() * 2);
			}
			state->initial_sol_update = state_variable.head(state->ndof());
			state->initial_vel_update = state_variable.tail(state->ndof());
		}
	}
	Eigen::VectorXd InitialConditionVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
				AdjointTools::dJ_initial_condition_adjoint_term(*state, state->get_adjoint_mat(1), state->get_adjoint_mat(0), cur_term);
			else
				log_and_throw_adjoint_error("[{}] Static simulation not supported!", name());

			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return apply_parametrization_jacobian(term, x);
	}
	Eigen::VectorXd InitialConditionVariableToSimulation::inverse_eval()
	{
		auto &state = *states_[0];
		Eigen::MatrixXd sol, vel;
		state.initial_solution(sol);
		state.initial_velocity(vel);

		Eigen::VectorXd x(sol.size() + vel.size());
		x << sol, vel;
		return x;
	}

	void DirichletVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		auto tensor_problem = std::dynamic_pointer_cast<polyfem::assembler::GenericTensorProblem>(states_[0]->problem);
		assert(dirichlet_boundaries_.size() > 0);
		int dim = states_[0]->mesh->dimension();
		int num_steps = indices.size() / dim;
		for (int i = 0; i < num_steps; ++i)
			for (const int &b : dirichlet_boundaries_)
				tensor_problem->update_dirichlet_boundary(b, indices(i * dim) + 1, state_variable.segment(i * dim, dim));

		logger().info("Current dirichlet boundary {} is {}.", dirichlet_boundaries_[0], state_variable.transpose());
	}

	Eigen::VectorXd DirichletVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
				AdjointTools::dJ_dirichlet_transient_adjoint_term(*state, state->get_adjoint_mat(1), state->get_adjoint_mat(0), cur_term);
			else
				log_and_throw_adjoint_error("[{}] Static dirichlet boundary optimization not supported!", name());

			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return apply_parametrization_jacobian(term, x);
	}
	std::string DirichletVariableToSimulation::variable_to_string(const Eigen::VectorXd &variable)
	{
		return "";
	}
	Eigen::VectorXd DirichletVariableToSimulation::inverse_eval()
	{
		assert(dirichlet_boundaries_.size() > 0);
		assert(states_.size() > 0);

		int dim = states_[0]->mesh->dimension();
		Eigen::VectorXd x;
		for (const auto &b : states_[0]->args["boundary_conditions"]["dirichlet_boundary"])
			if (b["id"].get<int>() == dirichlet_boundaries_[0])
			{
				auto value = b["value"];
				if (value[0].is_array())
				{
					if (!states_[0]->problem->is_time_dependent())
						log_and_throw_adjoint_error("Simulation must be time dependent for timestep wise dirichlet.");
					Eigen::MatrixXd dirichlet = value;
					x.setZero(dirichlet.rows() * (dirichlet.cols() - 1));
					for (int j = 1; j < dirichlet.cols(); ++j)
						x.segment((j - 1) * dim, dim) = dirichlet.col(j);
				}
				else if (value[0].is_number())
				{
					if (states_[0]->problem->is_time_dependent())
						log_and_throw_adjoint_error("Simulation must be quasistatic for single value dirichlet.");
					x.resize(dim);
					x = value;
				}
				else if (value.is_string())
					assert(false);
				break;
			}

		return parametrization_.inverse_eval(x);
	}

	void PressureVariableToSimulation::update_state(const Eigen::VectorXd &state_variable, const Eigen::VectorXi &indices)
	{
		auto tensor_problem = std::dynamic_pointer_cast<polyfem::assembler::GenericTensorProblem>(states_[0]->problem);
		assert(pressure_boundaries_.size() > 0);
		for (int i = 0; i < indices.size(); ++i)
			for (const int &b : pressure_boundaries_)
				tensor_problem->update_pressure_boundary(b, indices(i) + 1, state_variable(i));

		logger().info("Current pressure boundary {} is {}.", pressure_boundaries_[0], state_variable.transpose());
	}

	Eigen::VectorXd PressureVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (auto state : states_)
		{
			if (state->problem->is_time_dependent())
			{
				Eigen::MatrixXd adjoint_nu, adjoint_p;
				adjoint_nu = state->get_adjoint_mat(1);
				adjoint_p = state->get_adjoint_mat(0);
				AdjointTools::dJ_pressure_transient_adjoint_term(*state, pressure_boundaries_, adjoint_nu, adjoint_p, cur_term);
			}
			else
			{
				AdjointTools::dJ_pressure_static_adjoint_term(*state, pressure_boundaries_, state->diff_cached.u(0), state->get_adjoint_mat(0), cur_term);
			}
			if (term.size() != cur_term.size())
				term = cur_term;
			else
				term += cur_term;
		}
		return apply_parametrization_jacobian(term, x);
	}

	std::string PressureVariableToSimulation::variable_to_string(const Eigen::VectorXd &variable)
	{
		return "";
	}

	Eigen::VectorXd PressureVariableToSimulation::inverse_eval()
	{
		assert(pressure_boundaries_.size() > 0);
		assert(states_.size() > 0);

		Eigen::VectorXd x;
		for (const auto &b : states_[0]->args["boundary_conditions"]["pressure_boundary"])
			if (b["id"].get<int>() == pressure_boundaries_[0])
			{
				auto value = b["value"];
				if (value.is_array())
				{
					if (!states_[0]->problem->is_time_dependent())
						log_and_throw_adjoint_error("Simulation must be time dependent for timestep wise pressure.");
					Eigen::VectorXd pressures = value;
					x = pressures.segment(1, pressures.size() - 1);
				}
				else if (value.is_number())
				{
					if (states_[0]->problem->is_time_dependent())
						log_and_throw_adjoint_error("Simulation must be quasistatic for single value pressure.");
					x.resize(1);
					x(0) = value;
				}
				else if (value.is_string())
					assert(false);
				break;
			}

		return parametrization_.inverse_eval(x);
	}
} // namespace polyfem::solver
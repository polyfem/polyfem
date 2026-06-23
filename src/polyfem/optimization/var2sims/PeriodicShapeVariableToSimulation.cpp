#include <polyfem/optimization/var2sims/PeriodicShapeVariableToSimulation.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/legacy/State.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/StateDiff.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/Core>

#include <cassert>
#include <string>
#include <utility>

namespace polyfem::solver
{
	PeriodicShapeVariableToSimulation::PeriodicShapeVariableToSimulation(
		StatePtrs states,
		DiffCachePtrs diff_caches,
		CompositeParametrization parametrizations)
		: dim_(states[0]->mesh->dimension()),
		  vertex_num_(states[0]->mesh->n_vertices()),
		  states_(std::move(states)),
		  diff_caches_(std::move(diff_caches)),
		  parametrization_(std::move(parametrizations))
	{
		assert(!states_.empty());
		assert(states_.size() == diff_caches_.size());

		for (const auto &s : states_)
		{
			if (s->mesh->dimension() != dim_)
			{
				log_and_throw_adjoint_error("Fail to construct periodic shape variable to simulation. Reason: mesh dimension mismatch between states.");
			}
			if (s->mesh->n_vertices() != vertex_num_)
			{
				log_and_throw_adjoint_error("Fail to construct periodic shape variable to simulation. Reason: mesh vertex num mismatch between states.");
			}
			if (s->problem->is_time_dependent())
			{
				log_and_throw_adjoint_error("Fail to construct periodic shape variable to simulation. Reason: transient simulations are not supported.");
			}
			if (!s->has_periodic_bc() || s->periodic_bc == nullptr)
			{
				log_and_throw_adjoint_error("Fail to construct periodic shape variable to simulation. Reason: periodic boundary conditions are not enabled.");
			}
			if (!s->periodic_bc->all_direction_periodic())
			{
				log_and_throw_adjoint_error("Fail to construct periodic shape variable to simulation. Reason: partial periodicity is not supported.");
			}
			if (!s->is_homogenization())
			{
				log_and_throw_adjoint_error("Fail to construct periodic shape variable to simulation. Reason: only homogenization problems are supported.");
			}
		}

		Eigen::MatrixXd V;
		states_[0]->get_vertices(V);
		periodic_mesh_map_ = std::make_unique<PeriodicMeshToMesh>(V);
	}

	std::string PeriodicShapeVariableToSimulation::name() const
	{
		return "periodic-shape";
	}

	ParameterType PeriodicShapeVariableToSimulation::parameter_type() const
	{
		return ParameterType::PeriodicShape;
	}

	bool PeriodicShapeVariableToSimulation::affect_state(const legacy::State &target) const
	{
		for (auto &s : states_)
		{
			if (s.get() == &target)
				return true;
		}
		return false;
	}

	void PeriodicShapeVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		Eigen::MatrixXd V = utils::unflatten(periodic_mesh_map_->eval(y), dim_);

		for (auto &s : states_)
		{
			for (int i = 0; i < vertex_num_; ++i)
				s->mesh->set_point(i, V.row(i));
		}
	}

	void PeriodicShapeVariableToSimulation::update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const
	{
		assert(state_variables.size() == para_out_dof());
		state_variables = parametrization_.eval(x);
	}

	Eigen::VectorXd PeriodicShapeVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		Eigen::VectorXd term, cur_term;
		for (int i = 0; i < states_.size(); ++i)
		{
			auto &state = states_[i];
			auto &diff_cache = diff_caches_[i];

			Eigen::MatrixXd adjoint_p = get_adjoint_mat(*state, *diff_cache, 0);

			AdjointTools::dJ_periodic_shape_adjoint_term(
				*state,
				*diff_cache,
				*periodic_mesh_map_,
				y,
				diff_cache->u(0),
				adjoint_p,
				cur_term);

			if (term.size() != cur_term.size())
			{
				term = cur_term;
			}
			else
			{
				term += cur_term;
			}
		}

		assert(term.size() == para_out_dof());
		return parametrization_.apply_jacobian(term, x);
	}

	int PeriodicShapeVariableToSimulation::inverse_dof() const
	{
		return parametrization_.inverse_size(para_out_dof());
	}

	Eigen::VectorXd PeriodicShapeVariableToSimulation::inverse_eval() const
	{
		Eigen::MatrixXd V;
		states_[0]->get_vertices(V);

		Eigen::VectorXd y = periodic_mesh_map_->inverse_eval(utils::flatten(V));
		return parametrization_.inverse_eval(y);
	}

	Eigen::VectorXd PeriodicShapeVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const
	{
		assert(term.size() == vertex_num_ * dim_);

		const Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		const Eigen::VectorXd reduced_term = periodic_mesh_map_->apply_jacobian(term, y);
		assert(reduced_term.size() == para_out_dof());

		return parametrization_.apply_jacobian(reduced_term, x);
	}

	int PeriodicShapeVariableToSimulation::para_out_dof() const
	{
		return periodic_mesh_map_->input_size();
	}

} // namespace polyfem::solver

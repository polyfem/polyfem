#include <polyfem/optimization/var2sims/DirichletNodesVariableToSimulation.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/legacy/State.hpp>
#include <polyfem/assembler/GenericProblem.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/StateDiff.hpp>
#include <polyfem/optimization/var2sims/ActiveSelectionUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <Eigen/Core>

#include <cassert>
#include <string>
#include <utility>
#include <vector>

namespace polyfem::solver
{

	DirichletNodesVariableToSimulation::DirichletNodesVariableToSimulation(
		StatePtrs states,
		DiffCachePtrs diff_caches,
		CompositeParametrization parametrizations,
		Eigen::VectorXi active_geom_nodes)
		: dim_(states[0]->mesh->dimension()),
		  vertex_num_(states[0]->mesh->n_vertices()),
		  states_(std::move(states)),
		  diff_caches_(std::move(diff_caches)),
		  parametrization_(std::move(parametrizations)),
		  active_geom_nodes_(std::move(active_geom_nodes))
	{
		assert(!states_.empty());
		assert(states_.size() == diff_caches_.size());

		// Quasistatic only.
		for (auto &s : states_)
		{
			if (s->problem->is_time_dependent())
			{
				log_and_throw_adjoint_error("Fail to construct dirichlet nodes variable to simulation. Reason: only quasistatic simulations supported.");
			}
		}

		// Expand implicit all-active node selection.
		if (active_geom_nodes_.size() == 0)
		{
			auto &s = *states_[0];
			std::vector<int> tmp;
			for (int v_in = 0; v_in < vertex_num_; ++v_in)
			{
				int v = s.in_node_to_node(v_in);
				int tag = s.mesh->get_node_id(v);
				if (s.problem->is_nodal_dirichlet_boundary(v, tag))
				{
					tmp.push_back(v_in);
				}
			}
			std::sort(tmp.begin(), tmp.end());
			active_geom_nodes_ = Eigen::Map<Eigen::VectorXi>(tmp.data(), tmp.size());
		}

		// Validate the expanded node selection against every state.
		std::string reason;
		if (!is_active_dirichlet_node_valid(active_geom_nodes_, states_, reason))
		{
			log_and_throw_adjoint_error("Fail to construct dirichlet nodes variable to simulation. Reason: {}", reason);
		}
	}

	std::string DirichletNodesVariableToSimulation::name() const
	{
		return "dirichlet-nodes";
	}

	ParameterType DirichletNodesVariableToSimulation::parameter_type() const
	{
		return ParameterType::DirichletBC;
	}

	bool DirichletNodesVariableToSimulation::affect_state(const legacy::State &target) const
	{
		for (auto &s : states_)
		{
			if (s.get() == &target)
			{
				return true;
			}
		}
		return false;
	}

	void DirichletNodesVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		Eigen::MatrixXd nodal_dirichlet = utils::unflatten(y, dim_);

		for (auto &s : states_)
		{
			auto tensor_problem = std::dynamic_pointer_cast<polyfem::assembler::GenericTensorProblem>(s->problem);
			if (!tensor_problem)
			{
				log_and_throw_adjoint_error("[{}] Only tensor problems are supported.", name());
			}

			tensor_problem->update_dirichlet_nodes(s->in_node_to_node, active_geom_nodes_, nodal_dirichlet);
		}
	}

	void DirichletNodesVariableToSimulation::update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const
	{
		assert(state_variables.size() == para_out_dof());
		state_variables = parametrization_.eval(x);
	}

	Eigen::VectorXd DirichletNodesVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term = Eigen::VectorXd::Zero(para_out_dof());

		for (int si = 0; si < states_.size(); ++si)
		{
			auto &state = states_[si];
			auto &diff_cache = diff_caches_[si];

			Eigen::MatrixXd adjoint_p = get_adjoint_mat(*state, *diff_cache, 0);

			Eigen::VectorXd full_term;
			AdjointTools::dJ_dirichlet_static_adjoint_term(*state, *diff_cache, adjoint_p, full_term);

			assert(full_term.size() == vertex_num_ * dim_);

			for (int i = 0; i < active_geom_nodes_.size(); ++i)
			{
				term.segment(i * dim_, dim_) += full_term.segment(active_geom_nodes_(i) * dim_, dim_);
			}
		}

		assert(term.size() == para_out_dof());
		return parametrization_.apply_jacobian(term, x);
	}

	int DirichletNodesVariableToSimulation::inverse_dof() const
	{
		return parametrization_.inverse_size(para_out_dof());
	}

	Eigen::VectorXd DirichletNodesVariableToSimulation::inverse_eval() const
	{
		logger().warn("Inverse eval is not implemented in {}; falling back to zero.", name());
		return parametrization_.inverse_eval(Eigen::VectorXd::Zero(para_out_dof()));
	}

	Eigen::VectorXd DirichletNodesVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &, const Eigen::VectorXd &) const
	{
		// Not implemented because there's no user
		log_and_throw_adjoint_error("apply_parametrization_jacobian is not implemented in {} variable to simulation.", name());
	}

	int DirichletNodesVariableToSimulation::para_out_dof() const
	{
		return active_geom_nodes_.size() * dim_;
	}
} // namespace polyfem::solver

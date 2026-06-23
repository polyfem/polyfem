#include <polyfem/optimization/var2sims/ShapeVariableToSimulation.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/StateDiff.hpp>
#include <polyfem/optimization/var2sims/ParameterType.hpp>
#include <polyfem/optimization/var2sims/ActiveSelectionUtils.hpp>

#include <Eigen/Core>

#include <string>
#include <cassert>

namespace polyfem::solver
{

	ShapeVariableToSimulation::ShapeVariableToSimulation(StatePtrs states,
														 DiffCachePtrs diff_caches,
														 CompositeParametrization parametrizations,
														 Eigen::VectorXi active_dimensions,
														 Eigen::VectorXi active_geom_nodes)
		: dim_(states[0]->mesh->dimension()),
		  vertex_num_(states[0]->mesh->n_vertices()),
		  states_(std::move(states)),
		  diff_caches_(std::move(diff_caches)),
		  parametrization_(std::move(parametrizations)),
		  active_dimensions_(std::move(active_dimensions)),
		  active_geom_nodes_(std::move(active_geom_nodes))
	{
		assert(!states_.empty());
		assert(states_.size() == diff_caches_.size());

		// Validates active selections.
		std::string reason;
		if (!is_active_dims_valid(active_dimensions_, states_, reason))
		{
			log_and_throw_adjoint_error("Fail to construct shape variable to simulation. Reason: {}", reason);
		}
		if (!is_active_geom_nodes_valid(active_geom_nodes_, states_, reason))
		{
			log_and_throw_adjoint_error("Fail to construct shape variable to simulation. Reason: {}", reason);
		}

		// Expand implicit all active selection.
		if (active_dimensions_.size() == 0)
		{
			active_dimensions_ = Eigen::VectorXi::LinSpaced(dim_, 0, dim_ - 1);
		}
		if (active_geom_nodes_.size() == 0)
		{
			active_geom_nodes_ = Eigen::VectorXi::LinSpaced(vertex_num_, 0, vertex_num_ - 1);
		}
	}

	std::string ShapeVariableToSimulation::name() const
	{
		return "shape";
	}

	ParameterType ShapeVariableToSimulation::parameter_type() const
	{
		return ParameterType::Shape;
	}

	bool ShapeVariableToSimulation::affect_state(const legacy::State &target) const
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

	void ShapeVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		int active_dim_num = active_dimensions_.size();
		for (auto &s : states_)
		{
			for (int ni = 0; ni < active_geom_nodes_.size(); ++ni)
			{
				int node_id = active_geom_nodes_(ni);
				Eigen::VectorXd p = s->mesh->point(node_id);
				for (int di = 0; di < active_dimensions_.size(); ++di)
				{
					int d = active_dimensions_(di);
					p(d) = y(ni * active_dim_num + di);
				}
				s->mesh->set_point(node_id, p);
			}
		}
	}

	void ShapeVariableToSimulation::update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const
	{
		assert(state_variables.size() == dim_ * vertex_num_);

		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		int active_dim_num = active_dimensions_.size();
		for (int ni = 0; ni < active_geom_nodes_.size(); ++ni)
		{
			int vertex_id = active_geom_nodes_(ni);
			for (int di = 0; di < active_dimensions_.size(); ++di)
			{
				int d = active_dimensions_(di);
				state_variables(vertex_id * dim_ + d) = y(ni * active_dim_num + di);
			}
		}
	}

	Eigen::VectorXd ShapeVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (int i = 0; i < states_.size(); ++i)
		{
			auto &state = states_[i];
			auto &diff_cache = diff_caches_[i];

			if (state->problem->is_time_dependent())
			{
				Eigen::MatrixXd adjoint_p = get_adjoint_mat(*state, *diff_cache, 0);
				Eigen::MatrixXd adjoint_nu = get_adjoint_mat(*state, *diff_cache, 1);
				AdjointTools::dJ_shape_transient_adjoint_term(*state, *diff_cache, adjoint_nu, adjoint_p, cur_term);
			}
			else if (state->is_homogenization())
			{
				Eigen::MatrixXd adjoint_p = get_adjoint_mat(*state, *diff_cache, 0);
				AdjointTools::dJ_shape_homogenization_adjoint_term(*state,
																   *diff_cache,
																   diff_cache->u(0),
																   adjoint_p,
																   cur_term);
			}
			else
			{
				Eigen::MatrixXd adjoint_p = get_adjoint_mat(*state, *diff_cache, 0);
				AdjointTools::dJ_shape_static_adjoint_term(*state,
														   *diff_cache,
														   diff_cache->u(0),
														   adjoint_p,
														   cur_term);
			}

			if (term.size() != cur_term.size())
			{
				term = cur_term;
			}
			else
			{
				term += cur_term;
			}
		}

		assert(term.size() == vertex_num_ * dim_);

		Eigen::VectorXd active_term(para_out_dof());
		int active_dim_num = active_dimensions_.size();
		for (int i = 0; i < active_geom_nodes_.size(); ++i)
		{
			int vertex_id = active_geom_nodes_(i);
			for (int di = 0; di < active_dimensions_.size(); ++di)
			{
				int d = active_dimensions_(di);
				active_term(i * active_dim_num + di) = term(vertex_id * dim_ + d);
			}
		}

		assert(active_term.size() == para_out_dof());
		return parametrization_.apply_jacobian(active_term, x);
	}

	int ShapeVariableToSimulation::inverse_dof() const
	{
		return parametrization_.inverse_size(para_out_dof());
	}

	Eigen::VectorXd ShapeVariableToSimulation::inverse_eval() const
	{
		Eigen::VectorXd x = Eigen::VectorXd::Zero(para_out_dof());
		int active_dim_num = active_dimensions_.size();
		for (int i = 0; i < active_geom_nodes_.size(); ++i)
		{
			Eigen::VectorXd p = states_[0]->mesh->point(active_geom_nodes_(i));
			for (int di = 0; di < active_dimensions_.size(); ++di)
			{
				int d = active_dimensions_(di);
				x(i * active_dim_num + di) = p(d);
			}
		}

		return parametrization_.inverse_eval(x);
	}

	Eigen::VectorXd ShapeVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const
	{
		// Forms expect term to be full dof before any selection.
		assert(term.size() == vertex_num_ * dim_);

		Eigen::VectorXd active_term(para_out_dof());
		int active_dim_num = active_dimensions_.size();
		for (int i = 0; i < active_geom_nodes_.size(); ++i)
		{
			int vertex_id = active_geom_nodes_(i);
			for (int di = 0; di < active_dimensions_.size(); ++di)
			{
				int d = active_dimensions_(di);
				active_term(i * active_dim_num + di) = term(vertex_id * dim_ + d);
			}
		}
		return parametrization_.apply_jacobian(active_term, x);
	}

	int ShapeVariableToSimulation::para_out_dof() const
	{
		return active_dimensions_.size() * active_geom_nodes_.size();
	}

} // namespace polyfem::solver

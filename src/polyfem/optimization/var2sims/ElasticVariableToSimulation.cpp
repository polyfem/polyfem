#include <polyfem/optimization/var2sims/ElasticVariableToSimulation.hpp>

#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/StateDiff.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <Eigen/Core>

#include <string>
#include <cassert>

namespace polyfem::solver
{
	ElasticVariableToSimulation::ElasticVariableToSimulation(StatePtrs states,
															 DiffCachePtrs diff_caches,
															 CompositeParametrization parametrizations)
		: states_(std::move(states)),
		  diff_caches_(std::move(diff_caches)),
		  parametrization_(std::move(parametrizations))
	{
		assert(!states_.empty());
		assert(states_.size() == diff_caches_.size());

		elem_num_ = states_[0]->bases.size();
		for (auto &s : states_)
		{
			if (s->bases.size() != elem_num_)
			{
				log_and_throw_adjoint_error("Fail to construct elastic variable to simulation. Reason: Inconsistent element numbers.");
			}
		}
	}

	std::string ElasticVariableToSimulation::name() const
	{
		return "elastic";
	}

	ParameterType ElasticVariableToSimulation::parameter_type() const
	{
		return ParameterType::LameParameter;
	}

	bool ElasticVariableToSimulation::affect_state(const legacy::State &target) const
	{
		for (const auto &s : states_)
		{
			if (s.get() == &target)
			{
				return true;
			}
		}
		return false;
	}

	void ElasticVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		for (auto &s : states_)
		{
			s->assembler->update_lame_params(y.segment(0, elem_num_), y.segment(elem_num_, elem_num_));
		}
	}

	void ElasticVariableToSimulation::update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const
	{
		assert(state_variables.size() == para_out_dof());
		state_variables = parametrization_.eval(x);
	}

	Eigen::VectorXd ElasticVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (int i = 0; i < int(states_.size()); ++i)
		{
			auto &state = states_[i];
			auto &diff_cache = diff_caches_[i];

			if (state->problem->is_time_dependent())
			{
				Eigen::MatrixXd adjoint_p = get_adjoint_mat(*state, *diff_cache, 0);
				Eigen::MatrixXd adjoint_nu = get_adjoint_mat(*state, *diff_cache, 1);
				AdjointTools::dJ_material_transient_adjoint_term(*state, *diff_cache, adjoint_nu, adjoint_p, cur_term);
			}
			else
			{
				Eigen::MatrixXd adjoint_p = get_adjoint_mat(*state, *diff_cache, 0);
				AdjointTools::dJ_material_static_adjoint_term(*state, diff_cache->u(0), adjoint_p, cur_term);
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

		assert(term.size() == para_out_dof());
		return parametrization_.apply_jacobian(term, x);
	}

	int ElasticVariableToSimulation::inverse_dof() const
	{
		return parametrization_.inverse_size(para_out_dof());
	}

	Eigen::VectorXd ElasticVariableToSimulation::inverse_eval() const
	{
		// Sample lame parameters at barycenter of each mesh element.
		auto &state = *(states_[0]);
		auto params_map = state.assembler->parameters();

		// params_map returns (key, callback). Callback interpolates param usign FE basis.
		auto search_lambda = params_map.find("lambda");
		auto search_mu = params_map.find("mu");
		if (search_lambda == params_map.end() || search_mu == params_map.end())
		{
			log_and_throw_adjoint_error("[{}] Failed to find Lame parameters!", name());
		}

		int dim = state.mesh->dimension();
		Eigen::VectorXd lambdas(elem_num_);
		Eigen::VectorXd mus(elem_num_);
		for (int e = 0; e < elem_num_; ++e)
		{
			RowVectorNd barycenter;
			if (!state.mesh->is_volume())
			{
				auto mesh2d = dynamic_cast<const mesh::Mesh2D *>(state.mesh.get());
				barycenter = mesh2d->face_barycenter(e);
			}
			else
			{
				auto mesh3d = dynamic_cast<const mesh::Mesh3D *>(state.mesh.get());
				barycenter = mesh3d->cell_barycenter(e);
			}
			// The callback signature is (uv coordinate, world coordinate, time, elem id)
			// For lame parameters specifically, uv is not used so we pass a dummy.
			lambdas(e) = search_lambda->second(RowVectorNd::Zero(dim), barycenter, 0.0f, e);
			mus(e) = search_mu->second(RowVectorNd::Zero(dim), barycenter, 0.0f, e);
		}

		Eigen::VectorXd params(para_out_dof());
		params << lambdas, mus;
		return parametrization_.inverse_eval(params);
	}

	Eigen::VectorXd ElasticVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &x) const
	{
		assert(term.size() == 2 * elem_num_);
		return parametrization_.apply_jacobian(term, x);
	}

	int ElasticVariableToSimulation::para_out_dof() const
	{
		return 2 * elem_num_;
	}

} // namespace polyfem::solver

#include <polyfem/optimization/var2sims/FrictionVariableToSimulation.hpp>

#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/StateDiff.hpp>
#include <polyfem/optimization/var2sims/ParameterType.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/Core>

#include <string>
#include <cassert>

namespace polyfem::solver
{

	FrictionVariableToSimulation::FrictionVariableToSimulation(StatePtrs states,
															   DiffCachePtrs diff_caches,
															   CompositeParametrization parametrizations)
		: states_(std::move(states)),
		  diff_caches_(std::move(diff_caches)),
		  parametrization_(std::move(parametrizations))
	{
		assert(!states_.empty());
		assert(states_.size() == diff_caches_.size());

		for (auto &s : states_)
		{
			if (!s->problem->is_time_dependent())
			{
				log_and_throw_adjoint_error("Fail to construct friction variable to simulation. Reason: Can't optimize friction for static problem.");
			}
		}
	}

	std::string FrictionVariableToSimulation::name() const
	{
		return "friction";
	}

	ParameterType FrictionVariableToSimulation::parameter_type() const
	{
		return ParameterType::FrictionCoefficient;
	}

	bool FrictionVariableToSimulation::affect_state(const legacy::State &target) const
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

	void FrictionVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		for (auto &s : states_)
		{
			s->args["contact"]["friction_coefficient"] = y(0);
		}
	}

	void FrictionVariableToSimulation::update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const
	{
		assert(state_variables.size() == para_out_dof());
		state_variables = parametrization_.eval(x);
	}

	Eigen::VectorXd FrictionVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (int i = 0; i < int(states_.size()); ++i)
		{
			auto &state = states_[i];
			auto &diff_cache = diff_caches_[i];

			assert(state->problem->is_time_dependent());
			Eigen::MatrixXd adjoint_p = get_adjoint_mat(*state, *diff_cache, 0);
			Eigen::MatrixXd adjoint_nu = get_adjoint_mat(*state, *diff_cache, 1);
			AdjointTools::dJ_friction_transient_adjoint_term(*state, *diff_cache, adjoint_nu, adjoint_p, cur_term);

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

	int FrictionVariableToSimulation::inverse_dof() const
	{
		return parametrization_.inverse_size(para_out_dof());
	}

	Eigen::VectorXd FrictionVariableToSimulation::inverse_eval() const
	{
		Eigen::VectorXd y(para_out_dof());
		y(0) = states_[0]->args["contact"]["friction_coefficient"].get<double>();
		return parametrization_.inverse_eval(y);
	}

	Eigen::VectorXd FrictionVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &, const Eigen::VectorXd &) const
	{
		// Not implemented because there's no user
		log_and_throw_adjoint_error("apply_parametrization_jacobian is not implemented in {} variable to simulation.", name());
	}

	int FrictionVariableToSimulation::para_out_dof() const
	{
		return 1;
	}

} // namespace polyfem::solver

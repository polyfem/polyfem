#include <polyfem/optimization/var2sims/DampingVariableToSimulation.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/assembler/ViscousDamping.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/StateDiff.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/Core>

#include <cassert>
#include <string>

namespace polyfem::solver
{

	DampingVariableToSimulation::DampingVariableToSimulation(StatePtrs states,
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
				log_and_throw_adjoint_error("Fail to construct damping variable to simulation. Reason: Can't optimize damping for static problem.");
			}
		}
	}

	std::string DampingVariableToSimulation::name() const
	{
		return "damping";
	}

	ParameterType DampingVariableToSimulation::parameter_type() const
	{
		return ParameterType::DampingCoefficient;
	}

	bool DampingVariableToSimulation::affect_state(const legacy::State &target) const
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

	void DampingVariableToSimulation::update(const Eigen::VectorXd &x)
	{
		Eigen::VectorXd y = parametrization_.eval(x);
		assert(y.size() == para_out_dof());

		double psi = y(0);
		double phi = y(1);

		for (auto &s : states_)
		{
			if (!s->args["materials"].is_array())
			{
				s->args["materials"]["psi"] = psi;
				s->args["materials"]["phi"] = phi;
			}
			else
			{
				for (auto &arg : s->args["materials"])
				{
					arg["psi"] = psi;
					arg["phi"] = phi;
				}
			}

			if (s->damping_assembler)
			{
				json damping_param = {
					{"psi", psi},
					{"phi", phi},
				};
				s->damping_assembler->add_multimaterial(0, damping_param, s->units, s->root_path());
			}
		}
	}

	void DampingVariableToSimulation::update_state_variables(const Eigen::VectorXd &x, Eigen::VectorXd &state_variables) const
	{
		assert(state_variables.size() == para_out_dof());
		state_variables = parametrization_.eval(x);
	}

	Eigen::VectorXd DampingVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd term, cur_term;
		for (int i = 0; i < int(states_.size()); ++i)
		{
			auto &state = states_[i];
			auto &diff_cache = diff_caches_[i];

			assert(state->problem->is_time_dependent());
			Eigen::MatrixXd adjoint_p = get_adjoint_mat(*state, *diff_cache, 0);
			Eigen::MatrixXd adjoint_nu = get_adjoint_mat(*state, *diff_cache, 1);
			AdjointTools::dJ_damping_transient_adjoint_term(*state, *diff_cache, adjoint_nu, adjoint_p, cur_term);

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

	int DampingVariableToSimulation::inverse_dof() const
	{
		return parametrization_.inverse_size(para_out_dof());
	}

	Eigen::VectorXd DampingVariableToSimulation::inverse_eval() const
	{
		Eigen::VectorXd y(para_out_dof());
		json material = states_[0]->args["materials"];
		if (material.is_array())
		{
			material = material[0];
		}

		y(0) = material["psi"].get<double>();
		y(1) = material["phi"].get<double>();
		return parametrization_.inverse_eval(y);
	}

	Eigen::VectorXd DampingVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &, const Eigen::VectorXd &) const
	{
		// Not implemented because there's no user
		log_and_throw_adjoint_error("apply_parametrization_jacobian is not implemented in {} variable to simulation.", name());
	}

	int DampingVariableToSimulation::para_out_dof() const
	{
		return 2;
	}

} // namespace polyfem::solver

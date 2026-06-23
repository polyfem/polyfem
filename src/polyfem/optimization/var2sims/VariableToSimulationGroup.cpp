#include <polyfem/optimization/var2sims/VariableToSimulationGroup.hpp>

#include <polyfem/optimization/var2sims/VariableToSimulation.hpp>
#include <polyfem/optimization/var2sims/ParameterType.hpp>

namespace polyfem::solver
{

	void VariableToSimulationGroup::update(const Eigen::VectorXd &x)
	{
		for (auto &v2s : data)
		{
			v2s->update(x);
		}
	}

	void VariableToSimulationGroup::compute_state_variable(const ParameterType type,
														   const legacy::State &target,
														   const Eigen::VectorXd &x,
														   Eigen::VectorXd &state_variable) const
	{

		for (const auto &v2s : data)
		{
			if (v2s->parameter_type() != type)
			{
				continue;
			}
			if (!v2s->affect_state(target))
			{
				continue;
			}

			// If multiple var2sim updates the same dof, the later will overwrite previous updates.
			v2s->update_state_variables(x, state_variable);
		}
	}

	Eigen::VectorXd VariableToSimulationGroup::compute_adjoint_term(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd adjoint_term = Eigen::VectorXd::Zero(x.size());
		for (const auto &v2s : data)
		{
			adjoint_term += v2s->compute_adjoint_term(x);
		}
		return adjoint_term;
	}

	Eigen::VectorXd VariableToSimulationGroup::apply_parametrization_jacobian(ParameterType type,
																			  const legacy::State &target,
																			  const Eigen::VectorXd &x,
																			  const std::function<Eigen::VectorXd()> &grad) const
	{
		Eigen::VectorXd gradv = Eigen::VectorXd::Zero(x.size());
		for (const auto &v2s : data)
		{
			if (v2s->parameter_type() != type)
			{
				continue;
			}
			if (!v2s->affect_state(target))
			{
				continue;
			}

			gradv += v2s->apply_parametrization_jacobian(grad(), x);
		}
		return gradv;
	}

} // namespace polyfem::solver

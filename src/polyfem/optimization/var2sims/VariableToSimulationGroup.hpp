#pragma once

#include <polyfem/optimization/var2sims/VariableToSimulation.hpp>
#include <polyfem/optimization/var2sims/ParameterType.hpp>

#include <functional>

namespace polyfem::solver
{
	class VariableToSimulationGroup
	{
	public:
		void update(const Eigen::VectorXd &x);

		void compute_state_variable(ParameterType type,
									const legacy::State &target,
									const Eigen::VectorXd &x,
									Eigen::VectorXd &state_variable) const;

		Eigen::VectorXd compute_adjoint_term(const Eigen::VectorXd &x) const;

		/// @brief Compute parametrization jacobian for all var2sim matching
		/// parameter type and output to target state.
		Eigen::VectorXd apply_parametrization_jacobian(ParameterType type,
													   const legacy::State &target,
													   const Eigen::VectorXd &x,
													   const std::function<Eigen::VectorXd()> &grad) const;

		std::vector<std::shared_ptr<VariableToSimulation>> data;
	};

} // namespace polyfem::solver

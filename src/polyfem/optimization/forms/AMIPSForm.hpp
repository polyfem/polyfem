#pragma once

#include <polyfem/optimization/forms/VariableToSimulation.hpp>
#include "VariableToSimulation.hpp"
#include "AdjointForm.hpp"
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/basis/ElementBases.hpp>

namespace polyfem
{
	class State;
	namespace assembler
	{
		class Assembler;
	}
} // namespace polyfem

namespace polyfem::solver
{
	/// @brief Compute the minimum jacobian of the mesh elements, not differentiable
	/// polygon elements not supported!
	class MinJacobianForm : public AdjointForm
	{
	public:
		MinJacobianForm(const VariableToSimulationGroup &variable_to_simulation, const State &state)
			: AdjointForm(variable_to_simulation),
			  state_(state)
		{
		}

		virtual std::string name() const override { return "min-jacobian"; }

		double value_unweighted(const Eigen::VectorXd &x) const override;
		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	private:
		const State &state_;
	};

	class AMIPSForm : public AdjointForm
	{
	public:
		AMIPSForm(const VariableToSimulationGroup &variable_to_simulation, const State &state);

		virtual std::string name() const override { return "AMIPS"; }

		double value_unweighted(const Eigen::VectorXd &x) const override;
		void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

	private:
		Eigen::VectorXd get_updated_mesh_nodes(const Eigen::VectorXd &x) const
		{
			Eigen::VectorXd X = X_rest;
			variable_to_simulations_.compute_state_variable(ParameterType::Shape, &state_, x, X);
			return X;
		}

		const State &state_;

		Eigen::VectorXd X_rest;
		Eigen::MatrixXi F;
		std::vector<polyfem::basis::ElementBases> init_geom_bases_;
		assembler::AssemblyValsCache init_ass_vals_cache_;
		std::shared_ptr<assembler::Assembler> amips_energy_;
	};
} // namespace polyfem::solver

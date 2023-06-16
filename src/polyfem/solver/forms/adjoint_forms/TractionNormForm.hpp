#pragma once

#include "SpatialIntegralForms.hpp"
#include <ipc/collisions/collision_constraints.hpp>
#include <ipc/friction/friction_constraints.hpp>
#include <ipc/collision_mesh.hpp>
#include <ipc/broad_phase/broad_phase.hpp>

namespace polyfem::solver
{
	class TractionNormForm : public SpatialIntegralForm
	{
	public:
		TractionNormForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : SpatialIntegralForm(variable_to_simulations, state, args)
		{
			set_integral_type(SpatialIntegralType::SURFACE);

			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

			if (args["power"] > 0)
				in_power_ = args["power"];
		}

	protected:
		IntegrableFunctional get_integral_functional() const override;

	private:
		int in_power_ = 2;
	};

	class ContactForceForm : public StaticForm
	{
	public:
		ContactForceForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const State &state, const json &args) : StaticForm(variable_to_simulations), state_(state)
		{
			auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
			ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

			build_active_nodes();
		}
		~ContactForceForm() = default;

		const State &get_state() { return state_; }

		double value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const override;
		Eigen::VectorXd compute_adjoint_rhs_unweighted_step(const int time_step, const Eigen::VectorXd &x, const State &state) const override;
		Eigen::VectorXd compute_adjoint_rhs_unweighted_step_prev(const int time_step, const Eigen::VectorXd &x, const State &state) const override;
		virtual void compute_partial_gradient_unweighted_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		void build_active_nodes();

	protected:
		const State &state_;
		std::set<int> ids_;
		Eigen::VectorXi active_nodes_;
		StiffnessMatrix active_nodes_mat_;
		int dim_;

		double dhat_, epsv_, barrier_stiffness_, friction_coefficient_;
	};
} // namespace polyfem::solver
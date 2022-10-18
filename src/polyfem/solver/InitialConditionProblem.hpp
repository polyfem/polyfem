#pragma once

#include <polyfem/solver/OptimizationProblem.hpp>

namespace polyfem
{
	class InitialConditionProblem : public OptimizationProblem
	{
	public:
		InitialConditionProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_);

		using OptimizationProblem::gradient;
		using OptimizationProblem::value;

		double value(const TVector &x) override;
		double target_value(const TVector &x) override { return j->energy(state); }
		void gradient(const TVector &x, TVector &gradv) override { target_gradient(x, gradv); }
		void target_gradient(const TVector &x, TVector &gradv) override;

		bool is_step_valid(const TVector &x0, const TVector &x1) override { return (x1 - x0).cwiseAbs().maxCoeff() <= max_change; };
		bool is_step_collision_free(const TVector &x0, const TVector &x1) override { return true; }

		void line_search_end(bool failed) override;
		bool remesh(TVector &x) override { return false; }

		int optimization_dim() override { return 0; }

		bool solution_changed_pre(const TVector &newX) override;

		std::function<void(const TVector &x, Eigen::MatrixXd &init_sol, Eigen::MatrixXd &init_vel, State &state)> x_to_param;
		std::function<void(TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel, State &state)> param_to_x, dparam_to_dx;
	};
} // namespace polyfem

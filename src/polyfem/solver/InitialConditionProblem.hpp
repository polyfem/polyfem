#pragma once

#include <polyfem/solver/OptimizationProblem.hpp>

namespace polyfem
{
	class InitialConditionProblem : public OptimizationProblem
	{
	public:
		InitialConditionProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_, const json &args);

		double value(const TVector &x) override;
		double target_value(const TVector &x) { return j->energy(state); }
		void gradient(const TVector &x, TVector &gradv) override { target_gradient(x, gradv); }
		void target_gradient(const TVector &x, TVector &gradv);

		double value(const TVector &x, const bool only_elastic) { return value(x); };
		void gradient(const TVector &x, TVector &gradv, const bool only_elastic) { gradient(x, gradv); };

		bool is_step_valid(const TVector &x0, const TVector &x1) { return true; };
		bool is_step_collision_free(const TVector &x0, const TVector &x1) { return true; };
		double max_step_size(const TVector &x0, const TVector &x1) { return 1.; };

		void line_search_end(bool failed);
		bool remesh(TVector &x) { return false; };

		void solution_changed(const TVector &newX) override;

		std::function<void(const TVector &x, Eigen::MatrixXd &init_sol, Eigen::MatrixXd &init_vel)> x_to_param;
		std::function<void(TVector &x, const Eigen::MatrixXd &init_sol, const Eigen::MatrixXd &init_vel)> param_to_x, dparam_to_dx;
	};
} // namespace polyfem

#pragma once

#include <polyfem/solver/OptimizationProblem.hpp>

namespace polyfem
{
	class ControlProblem : public OptimizationProblem
	{
	public:
		ControlProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_);

		double target_value(const TVector &x) { return target_weight * j->energy(state); }
		double smooth_value(const TVector &x);

		void target_gradient(const TVector &x, TVector &gradv);
		void smooth_gradient(const TVector &x, TVector &gradv);

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;

		double value(const TVector &x, const bool only_elastic) { return value(x); };
		void gradient(const TVector &x, TVector &gradv, const bool only_elastic) { gradient(x, gradv); };

		bool is_step_valid(const TVector &x0, const TVector &x1);
		bool is_step_collision_free(const TVector &x0, const TVector &x1) { return true; };
		TVector force_inequality_constraint(const TVector &x0, const TVector &dx) { return x0 + dx; }
		double max_step_size(const TVector &x0, const TVector &x1) { return 1; }
		double heuristic_max_step(const TVector &dx) { return opt_params.contains("max_step") ? opt_params["max_step"].get<double>() : 1; };

		void line_search_end(bool failed);
		bool remesh(TVector &x) { return false; };

		bool solution_changed_pre(const TVector &newX) override;

		const std::map<int, int> &get_optimize_boundary_ids_to_position() { return optimize_boundary_ids_to_position; }

		// map x (optimization variables) to parameters (dirichlet at step i)
		std::function<void(const TVector &x, TVector &param)> x_to_param;
		// map parameters to x
		std::function<void(TVector &x, const TVector &param)> param_to_x;
		// compute gradient wrt. x given: gradient wrt. parameters, values of parameters
		std::function<void(TVector &dx, const TVector &dparams)> dparam_to_dx;

	private:
		double target_weight = 1;
		double smoothing_weight;
		std::vector<int> boundary_ids_list;

		std::map<int, int> optimize_boundary_ids_to_position;
	};
} // namespace polyfem

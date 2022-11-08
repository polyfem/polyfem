#pragma once

#include <polyfem/solver/OptimizationProblem.hpp>

namespace polyfem
{
	class ControlProblem : public OptimizationProblem
	{
	public:
		ControlProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_);

		using OptimizationProblem::gradient;
		using OptimizationProblem::value;

		double target_value(const TVector &x) override { return target_weight * j->energy(state); }
		double smooth_value(const TVector &x);

		void target_gradient(const TVector &x, TVector &gradv) override;
		void smooth_gradient(const TVector &x, TVector &gradv);

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;

		bool is_step_valid(const TVector &x0, const TVector &x1) override;
		bool is_step_collision_free(const TVector &x0, const TVector &x1) override { return true; };

		void line_search_end() override;
		bool remesh(TVector &x) override { return false; };
		int optimization_dim() override { return time_steps * optimize_boundary_ids_to_position.size() * state.mesh->dimension(); }

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

		int time_steps;

		json control_params;
		json smoothing_params;

		std::map<int, int> optimize_boundary_ids_to_position;
	};
} // namespace polyfem

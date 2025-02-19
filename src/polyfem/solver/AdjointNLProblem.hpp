#pragma once

#include <memory>
#include <polyfem/Common.hpp>
#include "FullNLProblem.hpp"
#include <polyfem/solver/forms/adjoint_forms/VariableToSimulation.hpp>
#include <fstream>

namespace polyfem
{
	class State;
}

namespace polyfem::solver
{
	class AdjointForm;

	class AdjointNLProblem : public FullNLProblem
	{
	public:
		AdjointNLProblem(std::shared_ptr<AdjointForm> form, const VariableToSimulationGroup &variables_to_simulation, const std::vector<std::shared_ptr<State>> &all_states, const json &args);
		AdjointNLProblem(std::shared_ptr<AdjointForm> form, const std::vector<std::shared_ptr<AdjointForm>> &stopping_conditions, const VariableToSimulationGroup &variables_to_simulation, const std::vector<std::shared_ptr<State>> &all_states, const json &args);
		virtual ~AdjointNLProblem() = default;

		double value(const Eigen::VectorXd &x) override;

		void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;
		void hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;
		void save_to_file(const int iter_num, const Eigen::VectorXd &x0);
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		void line_search_end() override;
		void post_step(const polysolve::nonlinear::PostStepData &data) override;
		bool stop(const TVector &x) override;

		void solution_changed(const Eigen::VectorXd &new_x) override;
		bool after_line_search_custom_operation(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		void solve_pde();

	private:
		std::shared_ptr<AdjointForm> form_;
		const VariableToSimulationGroup variables_to_simulation_;
		std::vector<std::shared_ptr<State>> all_states_;
		std::vector<bool> active_state_mask;
		Eigen::VectorXd cur_grad;
		Eigen::VectorXd curr_x;

		const int save_freq;
		std::ofstream solution_ostream;

		const bool enable_slim;
		const bool smooth_line_search;

		const bool solve_in_parallel;
		std::vector<int> solve_in_order;

		int save_iter = 0;

		std::vector<std::shared_ptr<AdjointForm>> stopping_conditions_; // if all the stopping conditions are non-positive, stop the optimization
	};
} // namespace polyfem::solver

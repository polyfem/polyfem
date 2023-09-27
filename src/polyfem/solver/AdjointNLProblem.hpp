#pragma once

#include <memory>
#include <polyfem/Common.hpp>
#include "FullNLProblem.hpp"

namespace polyfem
{
	class State;
}

namespace polyfem::solver
{
	class AdjointForm;
	class CompositeForm;
	class VariableToSimulation;

	class AdjointNLProblem : public FullNLProblem
	{
	public:
		AdjointNLProblem(std::shared_ptr<CompositeForm> composite_form, const std::vector<std::shared_ptr<VariableToSimulation>> &variables_to_simulation, const std::vector<std::shared_ptr<State>> &all_states, const json &args);
		AdjointNLProblem(std::shared_ptr<CompositeForm> composite_form, const std::vector<std::shared_ptr<AdjointForm>> stopping_conditions, const std::vector<std::shared_ptr<VariableToSimulation>> &variables_to_simulation, const std::vector<std::shared_ptr<State>> &all_states, const json &args);

		double value(const Eigen::VectorXd &x) override;

		void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;
		void hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;
		void save_to_file(const Eigen::VectorXd &x0) override;
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		void line_search_end() override;
		void post_step(const int iter_num, const Eigen::VectorXd &x) override;
		bool stop(const TVector &x) override;

		// virtual void set_project_to_psd(bool val) override;

		void solution_changed(const Eigen::VectorXd &new_x) override;

		void solve_pde();

		int n_states() const { return all_states_.size(); }
		std::shared_ptr<State> get_state(int id) { return all_states_[id]; }

	private:
		std::shared_ptr<CompositeForm> composite_form_;
		std::vector<std::shared_ptr<VariableToSimulation>> variables_to_simulation_;
		std::vector<std::shared_ptr<State>> all_states_;
		std::vector<bool> active_state_mask;
		Eigen::VectorXd cur_grad;
		int iter = 0;

		const int solve_log_level;
		const int save_freq;

		const bool solve_in_parallel;
		std::vector<int> solve_in_order;
		const bool better_initial_guess;

		std::vector<std::shared_ptr<AdjointForm>> stopping_conditions_; // if all the stopping conditions are non-positive, stop the optimization
	};
} // namespace polyfem::solver

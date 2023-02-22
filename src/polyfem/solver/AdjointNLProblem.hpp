#pragma once

#include <memory>
#include <polyfem/solver/forms/adjoint_forms/CompositeForm.hpp>
#include "FullNLProblem.hpp"
#include <polyfem/utils/Logger.hpp>
#include <polyfem/State.hpp>

namespace polyfem::solver
{
	class AdjointNLProblem : public FullNLProblem
	{
	public:
		AdjointNLProblem(std::shared_ptr<CompositeForm> composite_form, const std::vector<std::shared_ptr<VariableToSimulation>> &variables_to_simulation, const std::vector<std::shared_ptr<State>> &all_states, const json &args);

		double value(const Eigen::VectorXd &x) override;

		void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;
		inline void hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override
		{
			log_and_throw_error("Hessian not supported!");
		}

		Eigen::VectorXd component_values(const Eigen::VectorXd &x) override;
		Eigen::MatrixXd component_gradients(const Eigen::VectorXd &x) override;
		bool verify_gradient(const Eigen::VectorXd &x, const Eigen::VectorXd &gradv) override;

		bool smoothing(const Eigen::VectorXd &x, const Eigen::VectorXd &new_x, Eigen::VectorXd &smoothed_x) override;
		bool remesh(Eigen::VectorXd &x) override;
		void save_to_file(const Eigen::VectorXd &x0) override;

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		void line_search_end() override;
		void post_step(const int iter_num, const Eigen::VectorXd &x) override;

		// virtual void set_project_to_psd(bool val) override;

		void solution_changed(const Eigen::VectorXd &new_x) override;

		void solve_pde();

		int n_states() const { return all_states_.size(); }
		std::shared_ptr<State> get_state(int id) { return all_states_[id]; }

	private:
		std::shared_ptr<CompositeForm> composite_form_;
		std::vector<std::shared_ptr<State>> all_states_;
		std::vector<bool> active_state_mask;
		Eigen::VectorXd cur_grad;
		int iter = 0;

		const int solve_log_level;
		const int save_freq;

		const bool debug_finite_diff;
		const double finite_diff_eps;

		double adjoint_solve_time = 0;
		double grad_assembly_time = 0;

		Eigen::MatrixXd bounds_;

		std::vector<std::shared_ptr<VariableToSimulation>> variables_to_simulation_;
	};
} // namespace polyfem::solver

#pragma once

#include <cppoptlib/problem.h>
#include "HomoObjective.hpp"
#include "Parameter.hpp"

namespace polyfem::solver
{
	class AdjointNLProblem : public FullNLProblem
	{
	public:
		AdjointNLProblem(const std::shared_ptr<SumObjective> &obj, const std::vector<std::shared_ptr<Parameter>> &parameters, const std::vector<std::shared_ptr<State>> &all_states, const json &args)
			: obj_(obj),
			  parameters_(parameters),
			  all_states_(all_states),
			  better_initial_guess(args["solver"]["nonlinear"]["better_initial_guess"]),
			  solve_log_level(args["output"]["solve_log_level"]),
			  save_freq(args["output"]["save_frequency"]),
			  debug_finite_diff(args["solver"]["nonlinear"]["debug_fd"]),
			  finite_diff_eps(args["solver"]["nonlinear"]["debug_fd_eps"])
		{
			cur_x.setZero(0);
			cur_grad.setZero(0);
			for (const auto &p : parameters)
				optimization_dim_ += p->optimization_dim();

			active_state_mask.assign(all_states_.size(), false);
			int i = 0;
			for (const auto &state_ptr : all_states_)
			{
				for (const auto &p : parameters)
				{
					if (p->contains_state(*state_ptr))
					{
						active_state_mask[i] = true;
						break;
					}
				}
				i++;
			}
		}

		int full_size() const { return optimization_dim_; }

		double value(const Eigen::VectorXd &x) override;
		double value(const Eigen::VectorXd &x, const bool only_elastic);

		void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;
		void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv, const bool only_elastic);
		inline void hessian(const Eigen::VectorXd &x, const StiffnessMatrix &hessian) override
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

		virtual void set_project_to_psd(bool val) override;

		void solution_changed(const Eigen::VectorXd &new_x) override;

		void solve_pde();
		Eigen::VectorXd initial_guess() const;

		Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x) const;
		Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x) const;

		virtual int n_inequality_constraints();
		virtual double inequality_constraint_val(const Eigen::VectorXd &x, const int index);
		virtual Eigen::VectorXd inequality_constraint_grad(const Eigen::VectorXd &x, const int index);

		std::shared_ptr<State> get_state(int id) { return all_states_[id]; }

	private:
		int optimization_dim_ = 0;
		std::shared_ptr<SumObjective> obj_;
		std::vector<std::shared_ptr<Parameter>> parameters_;
		std::vector<std::shared_ptr<State>> all_states_;
		std::vector<bool> active_state_mask;
		Eigen::VectorXd cur_x, cur_grad;
		int iter = 0;

		const bool better_initial_guess;

		const int solve_log_level;
		const int save_freq;

		const bool debug_finite_diff;
		const double finite_diff_eps;

		double adjoint_solve_time = 0;
		double grad_assembly_time = 0;

		std::vector<VariableToSimulation> variables_to_simulation_;
	};
} // namespace polyfem::solver

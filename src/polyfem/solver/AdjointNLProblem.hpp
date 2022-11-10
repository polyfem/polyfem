#pragma once

#include <cppoptlib/problem.h>
#include "Objective.hpp"
#include "Parameter.hpp"

namespace polyfem::solver
{
	class AdjointNLProblem : public cppoptlib::Problem<double>
	{
	public:
		AdjointNLProblem(const std::shared_ptr<Objective> &obj, const std::vector<std::shared_ptr<Parameter>> &parameters, const std::vector<std::shared_ptr<State>> &all_states, const json &args) : obj_(obj), parameters_(parameters), all_states_(all_states), solve_log_level(args["output"]["solve_log_level"]), save_freq(args["output"]["save_frequency"])
		{
			for (const auto p : parameters)
				optimization_dim_ += p->optimization_dim();
		}

		int full_size() const { return optimization_dim_; }

		double target_value(const Eigen::VectorXd &x);
		double value(const Eigen::VectorXd &x) override;
		double value(const Eigen::VectorXd &x, const bool only_elastic);

		void target_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv);
		void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;
		void gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv, const bool only_elastic);

		void smoothing(const Eigen::VectorXd &x, Eigen::VectorXd &new_x);
		bool remesh(Eigen::VectorXd &x);

		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const;
		bool is_intersection_free(const Eigen::VectorXd &x) const;
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const;
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const;

		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1);
		void line_search_end();
		void post_step(const int iter_num, const Eigen::VectorXd &x);
		void save_to_file(const Eigen::VectorXd &x0);

		void solution_changed(const Eigen::VectorXd &new_x);

		Eigen::VectorXd get_lower_bound(const Eigen::VectorXd &x) const;
		Eigen::VectorXd get_upper_bound(const Eigen::VectorXd &x) const;

		int n_inequality_constraints();
		Eigen::VectorXd force_inequality_constraint(const Eigen::VectorXd &x0, const Eigen::VectorXd &dx);
		double inequality_constraint_val(const Eigen::VectorXd &x, const int index);
		Eigen::VectorXd inequality_constraint_grad(const Eigen::VectorXd &x, const int index);

	private:
		int optimization_dim_;
		std::shared_ptr<Objective> 				obj_;
		std::vector<std::shared_ptr<Parameter>> parameters_;
		std::vector<std::shared_ptr<State>>     all_states_;
		Eigen::VectorXd cur_x;
		int iter = 0;

		const int solve_log_level;
		const int save_freq;
	};
} // namespace polyfem::solver

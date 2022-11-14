#pragma once

#include <polyfem/State.hpp>
#include <polyfem/utils/CompositeFunctional.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <cppoptlib/problem.h>
#include <filesystem>

namespace polyfem
{
	class OptimizationProblem : public cppoptlib::Problem<double>
	{
	public:
		using typename cppoptlib::Problem<double>::Scalar;
		using typename cppoptlib::Problem<double>::TVector;

		OptimizationProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_);

		virtual ~OptimizationProblem() = default;

		virtual void solve_pde(const TVector &x) final;

		bool verify_gradient(const TVector &x, const TVector &gradv) { return true; }

		virtual double target_value(const TVector &x) = 0;
		virtual void target_gradient(const TVector &x, TVector &gradv) = 0;

		virtual double value(const TVector &x) = 0;
		virtual void gradient(const TVector &x, TVector &gradv) = 0;

		virtual double value(const TVector &x, const bool only_elastic) { return value(x); }
		virtual void gradient(const TVector &x, TVector &gradv, const bool only_elastic) { gradient(x, gradv); }

		virtual void smoothing(const TVector &x, TVector &new_x) {}
		virtual bool is_intersection_free(const TVector &x) { return true; }

		virtual void save_to_file(const TVector &x0) final;

		virtual bool stop(const TVector &x) { return false; }

		virtual int optimization_dim() = 0;

		virtual bool solution_changed_pre(const TVector &newX) = 0;

		virtual void solution_changed_post(const TVector &newX)
		{
			cur_x = newX;
			cur_grad.resize(0);
			cur_val = std::nan("");
		}

		virtual TVector get_lower_bound(const TVector &x) const
		{
			TVector min(x.size());
			min.setConstant(std::numeric_limits<double>::min());
			return min;
		}
		virtual TVector get_upper_bound(const TVector &x) const
		{
			TVector max(x.size());
			max.setConstant(std::numeric_limits<double>::max());
			return max;
		}

		virtual void solution_changed(const TVector &newX) final;

		virtual void post_step(const int iter_num, const TVector &x0) { iter++; }

		virtual void line_search_begin(const TVector &x0, const TVector &x1);

		virtual void line_search_end() = 0;

		virtual bool is_step_valid(const TVector &x0, const TVector &x1) = 0;

		virtual bool remesh(TVector &x) = 0;

		virtual TVector force_inequality_constraint(const TVector &x0, const TVector &dx) { return x0 + dx; }

		virtual double max_step_size(const TVector &x0, const TVector &x1) { return 1; }
		virtual bool is_step_collision_free(const TVector &x0, const TVector &x1) { return true; }

		virtual int n_inequality_constraints() { return 0; }
		virtual double inequality_constraint_val(const TVector &x, const int index)
		{
			assert(false);
			return std::nan("");
		}
		virtual TVector inequality_constraint_grad(const TVector &x, const int index)
		{
			assert(false);
			return TVector();
		}

	protected:
		State &state;
		std::string optimization_name = "";

		int iter = 0;
		int save_iter = -1;

		int dim;
		int actual_dim;

		int save_freq = 1;

		std::shared_ptr<CompositeFunctional> j;

		json opt_nonlinear_params;
		json opt_output_params;
		json opt_params;

		// better initial guess for forward solves
		Eigen::MatrixXd sol_at_ls_begin;
		TVector x_at_ls_begin;

		// store value and grad of current solution
		double cur_val;
		TVector cur_x, cur_grad;

		double max_change;
	};
} // namespace polyfem

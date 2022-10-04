#pragma once

#include <cppoptlib/problem.h>

namespace polyfem
{
	class GeneralOptimizationProblem : public cppoptlib::Problem<double>
	{
	public:
		using typename cppoptlib::Problem<double>::Scalar;
		using typename cppoptlib::Problem<double>::TVector;

		GeneralOptimizationProblem(std::vector<std::shared_ptr<OptimizationProblem>> subproblems_, const std::shared_ptr<CompositeFunctional> j_)
		{
			subproblems = subproblems_;
			for (const auto &subproblem : subproblems)
				optimization_dim_ += subproblem->optimization_dim();
		}

		int optimization_dim()  { return optimization_dim_; }

		double target_value(const TVector &x) 
		{
			double val = 0;
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				val += subproblem->target_value(x.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
			return val;
		}

		double value(const TVector &x) 
		{
			double val = 0;
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				val += subproblem->value(x.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
			return val;
		}

		double value(const TVector &x, const bool only_elastic) 
		{
			double val = 0;
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				val += subproblem->value(x.segment(cumulative, subproblem->optimization_dim()), only_elastic);
				cumulative += subproblem->optimization_dim();
			}
			return val;
		}

		void target_gradient(const TVector &x, TVector &gradv) 
		{
			gradv.resize(optimization_dim_);
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				TVector tmp;
				subproblem->target_gradient(x.segment(cumulative, subproblem->optimization_dim()), tmp);
				assert(tmp.size() == subproblem->optimization_dim());
				gradv.segment(cumulative, subproblem->optimization_dim()) = tmp;
				cumulative += subproblem->optimization_dim();
			}
		}

		void gradient(const TVector &x, TVector &gradv) 
		{
			gradv.resize(optimization_dim_);
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				TVector tmp;
				subproblem->gradient(x.segment(cumulative, subproblem->optimization_dim()), tmp);
				assert(tmp.size() == subproblem->optimization_dim());
				gradv.segment(cumulative, subproblem->optimization_dim()) = tmp;
				cumulative += subproblem->optimization_dim();
			}
		}

		void gradient(const TVector &x, TVector &gradv, const bool only_elastic) 
		{
			gradv.resize(optimization_dim_);
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				TVector tmp;
				subproblem->gradient(x.segment(cumulative, subproblem->optimization_dim()), tmp, only_elastic);
				assert(tmp.size() == subproblem->optimization_dim());
				gradv.segment(cumulative, subproblem->optimization_dim()) = tmp;
				cumulative += subproblem->optimization_dim();
			}
		}

		void smoothing(const TVector &x, TVector &new_x) 
		{
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				TVector tmp;
				tmp = new_x.segment(cumulative, subproblem->optimization_dim());
				subproblem->smoothing(x.segment(cumulative, subproblem->optimization_dim()), tmp);
				assert(tmp.size() == subproblem->optimization_dim());
				new_x.segment(cumulative, subproblem->optimization_dim()) = tmp;
				cumulative += subproblem->optimization_dim();
			}
		}

		bool is_step_valid(const TVector &x0, const TVector &x1) 
		{
			bool valid = true;
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				valid &= subproblem->is_step_valid(x0.segment(cumulative, subproblem->optimization_dim()), x1.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
			return valid;
		}

		bool is_intersection_free(const TVector &x)
		{
			bool intersection_free = true;
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				intersection_free &= subproblem->is_intersection_free(x.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
			return intersection_free;
		}

		bool is_step_collision_free(const TVector &x0, const TVector &x1)
		{
			bool collision_free = true;
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				collision_free &= subproblem->is_step_collision_free(x0.segment(cumulative, subproblem->optimization_dim()), x1.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
			return collision_free;
		}

		double max_step_size(const TVector &x0, const TVector &x1)
		{
			std::vector<double> step_sizes;
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				step_sizes.push_back(subproblem->max_step_size(x0.segment(cumulative, subproblem->optimization_dim()), x1.segment(cumulative, subproblem->optimization_dim())));
				cumulative += subproblem->optimization_dim();
			}
			auto min = std::min_element(step_sizes.begin(), step_sizes.end());
			assert(min != step_sizes.end());

			return *min;
		}

		bool remesh(TVector &x) 
		{
			bool remesh = false;
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				TVector tmp = x.segment(cumulative, subproblem->optimization_dim());
				remesh |= subproblem->remesh(tmp);
				assert(tmp.size() == subproblem->optimization_dim());
				x.segment(cumulative, subproblem->optimization_dim()) = tmp;
				cumulative += subproblem->optimization_dim();
			}
			return remesh;
		}

		void line_search_begin(const TVector &x0, const TVector &x1) 
		{

			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				subproblem->line_search_begin(x0.segment(cumulative, subproblem->optimization_dim()), x1.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
		}

		void line_search_end(bool failed) 
		{

			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				subproblem->line_search_end(failed);
				cumulative += subproblem->optimization_dim();
			}
		}

		void post_step(const int iter_num, const TVector &x0) 
		{
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				subproblem->post_step(iter_num, x0.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
			iter++;
		}

		bool solution_changed_pre(const TVector &newX) 
		{
			int cumulative = 0;
			bool flag = true;
			for (const auto &subproblem : subproblems)
			{
				flag &= subproblem->solution_changed_pre(newX.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
			return flag;
		}

		void solution_changed_post(const TVector &newX) 
		{
			cur_x = newX;
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				subproblem->solution_changed_post(newX.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
		}

		TVector get_lower_bound(const TVector &x) const
		{
			TVector min(x.size());
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				TVector tmp;
				min.segment(cumulative, subproblem->optimization_dim()) = subproblem->get_lower_bound(x.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
			return min;
		}
		TVector get_upper_bound(const TVector &x) const
		{
			TVector max(x.size());
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				TVector tmp;
				max.segment(cumulative, subproblem->optimization_dim()) = subproblem->get_upper_bound(x.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
			return max;
		}

		void solution_changed(const TVector &newX)
		{
			if (cur_x.size() == newX.size() && cur_x == newX)
				return;
			if (solution_changed_pre(newX))
			{
				int cumulative = 0;
				for (const auto &subproblem : subproblems)
				{
					subproblem->solve_pde(newX.segment(cumulative, subproblem->optimization_dim()));
					cumulative += subproblem->optimization_dim();
					break;
				}
			}
			solution_changed_post(newX);
		}

		TVector force_inequality_constraint(const TVector &x0, const TVector &dx)
		{
			TVector newX(x0.size());
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				newX.segment(cumulative, subproblem->optimization_dim()) = subproblem->force_inequality_constraint(x0.segment(cumulative, subproblem->optimization_dim()), dx.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
			return newX;
		}

		int n_inequality_constraints()
		{
			int num = 0;
			for (const auto &subproblem : subproblems)
			{
				num += subproblem->n_inequality_constraints();
			}
			return num;
		}

		double inequality_constraint_val(const TVector &x, const int index)
		{
			int num = 0;
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				num += subproblem->n_inequality_constraints();
				if (num > index)
				{
					num -= subproblem->n_inequality_constraints();
					return subproblem->inequality_constraint_val(x.segment(cumulative, subproblem->optimization_dim()), index - num);
				}
				cumulative += subproblem->optimization_dim();
			}
			log_and_throw_error("Exceeding number of inequality constraints!");
			return 0;
		}
		
		TVector inequality_constraint_grad(const TVector &x, const int index)
		{
			int num = 0;
			int cumulative = 0;
			TVector grad(x.size());
			grad.setZero();
			for (const auto &subproblem : subproblems)
			{
				num += subproblem->n_inequality_constraints();
				if (num > index)
				{
					num -= subproblem->n_inequality_constraints();
					grad.segment(cumulative, subproblem->optimization_dim()) = subproblem->inequality_constraint_grad(x.segment(cumulative, subproblem->optimization_dim()), index - num);
					break;
				}
				cumulative += subproblem->optimization_dim();
			}
			return grad;
		}

		void save_to_file(const TVector &x0)
		{
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				subproblem->save_to_file(x0.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
				break;
			}
		}

	private:
		std::vector<std::shared_ptr<OptimizationProblem>> subproblems;

		int iter = 0;
		int optimization_dim_ = 0;

		TVector cur_x;
	};
} // namespace polyfem

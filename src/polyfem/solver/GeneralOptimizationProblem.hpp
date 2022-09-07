#pragma once

#include "OptimizationProblem.hpp"

namespace polyfem
{
	class GeneralOptimizationProblem : public OptimizationProblem
	{
	public:
		GeneralOptimizationProblem(State &state_, std::vector<std::shared_ptr<OptimizationProblem>> subproblems_, const std::shared_ptr<CompositeFunctional> j_) : OptimizationProblem(state_, j_)
		{
			subproblems = subproblems_;
			for (const auto &subproblem : subproblems)
				optimization_dim_ += subproblem->optimization_dim();
		}

		int optimization_dim() override { return optimization_dim_; }

		double target_value(const TVector &x) override
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

		double value(const TVector &x) override
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

		double value(const TVector &x, const bool only_elastic) override
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

		void target_gradient(const TVector &x, TVector &gradv) override
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

		void gradient(const TVector &x, TVector &gradv) override
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

		void gradient(const TVector &x, TVector &gradv, const bool only_elastic) override
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

		void smoothing(const TVector &x, TVector &new_x) override
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

		bool is_step_valid(const TVector &x0, const TVector &x1) override
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

		bool remesh(TVector &x) override
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

		void line_search_begin(const TVector &x0, const TVector &x1) override
		{

			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				subproblem->line_search_begin(x0.segment(cumulative, subproblem->optimization_dim()), x1.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
		}

		void line_search_end(bool failed) override
		{

			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				subproblem->line_search_end(failed);
				cumulative += subproblem->optimization_dim();
			}
		}

		void post_step(const int iter_num, const TVector &x0) override
		{
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				subproblem->post_step(iter_num, x0.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
			iter++;
		}

		bool solution_changed_pre(const TVector &newX) override
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

		void solution_changed_post(const TVector &newX) override
		{
			int cumulative = 0;
			for (const auto &subproblem : subproblems)
			{
				subproblem->solution_changed_post(newX.segment(cumulative, subproblem->optimization_dim()));
				cumulative += subproblem->optimization_dim();
			}
		}

	private:
		std::vector<std::shared_ptr<OptimizationProblem>> subproblems;

		int optimization_dim_ = 0;
	};
} // namespace polyfem

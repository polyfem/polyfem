#pragma once

#include "OptimizationProblem.hpp"

namespace polyfem
{
	class TopologyOptimizationProblem : public OptimizationProblem
	{
	public:
		TopologyOptimizationProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_);

		double target_value(const TVector &x) { return j->energy(state); }
		double mass_value(const TVector &x);
		double smooth_value(const TVector &x);

		void target_gradient(const TVector &x, TVector &gradv);
		void mass_gradient(const TVector &x, TVector &gradv);
		void smooth_gradient(const TVector &x, TVector &gradv);

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;

		double value(const TVector &x, const bool only_elastic) { return value(x); };
		void gradient(const TVector &x, TVector &gradv, const bool only_elastic) { gradient(x, gradv); };

		bool is_step_valid(const TVector &x0, const TVector &x1);
		TVector force_inequality_constraint(const TVector &x0, const TVector &dx);
		bool is_intersection_free(const TVector &x) { return true; }
		bool is_step_collision_free(const TVector &x0, const TVector &x1) { return true; }
		double max_step_size(const TVector &x0, const TVector &x1);
		bool remesh(TVector &x) { return false; };

		TVector get_lower_bound(const TVector& x) 
		{
			TVector min(x.size());
			min.setConstant(min_density);
			return min; 
		}
		TVector get_upper_bound(const TVector& x) 
		{
			TVector max(x.size());
			max.setConstant(max_density);
			return max; 
		}

		void line_search_begin(const TVector &x0, const TVector &x1);
		void line_search_end(bool failed);

		void solution_changed(const TVector &newX) override;

		TVector apply_filter(const TVector &x);
		TVector apply_filter_to_grad(const TVector &x, const TVector &grad);
		
	private:
		double min_density = 0;
		double max_density = 1;

		bool has_mass_constraint;
		json mass_params;
		std::shared_ptr<CompositeFunctional> j_mass;

		bool has_smooth_constraint;
		json smooth_params;
		Eigen::SparseMatrix<bool> tt_adjacency;

		json top_params;

		bool has_filter;
		Eigen::SparseMatrix<double> tt_radius_adjacency;
		Eigen::VectorXd tt_radius_adjacency_row_sum;
	};
} // namespace polyfem

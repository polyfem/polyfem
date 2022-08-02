#pragma once

#include "OptimizationProblem.hpp"

namespace polyfem
{
	class TopologyOptimizationProblem : public OptimizationProblem
	{
	public:
		TopologyOptimizationProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_);

		double target_value(const TVector &x);
		double volume_value(const TVector &x);
		double smooth_value(const TVector &x);

		void target_gradient(const TVector &x, TVector &gradv);
		void volume_gradient(const TVector &x, TVector &gradv);
		void smooth_gradient(const TVector &x, TVector &gradv);

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;

		double value(const TVector &x, const bool only_elastic) { return value(x); };
		void gradient(const TVector &x, TVector &gradv, const bool only_elastic) { gradient(x, gradv); };

		void smoothing(const TVector &x, TVector &new_x) override;
		bool is_step_valid(const TVector &x0, const TVector &x1);
		bool is_intersection_free(const TVector &x);
		bool is_step_collision_free(const TVector &x0, const TVector &x1);
		double max_step_size(const TVector &x0, const TVector &x1);

		void line_search_begin(const TVector &x0, const TVector &x1);
		void line_search_end(bool failed);
		void post_step(const int iter_num, const TVector &x0) override;

		void solution_changed(const TVector &newX) override;

		void save_to_file(const TVector &x0) override;

	private:
		double min_density = 0;
		double max_density = 1;
	};
} // namespace polyfem

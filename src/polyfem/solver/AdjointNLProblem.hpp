#pragma once

#include <cppoptlib/problem.h>
#include "Objective.hpp"

namespace polyfem::solver
{
	class AdjointNLProblem : public cppoptlib::Problem<double>
	{
	public:
		using typename cppoptlib::Problem<double>::Scalar;
		using typename cppoptlib::Problem<double>::TVector;

		AdjointNLProblem(std::vector<std::shared_ptr<Objective>> &objectives): objectives_(objectives)
		{}

		double target_value(const TVector &x);
		double value(const TVector &x);
		double value(const TVector &x, const bool only_elastic);

		void target_gradient(const TVector &x, TVector &gradv);
		void gradient(const TVector &x, TVector &gradv);
		void gradient(const TVector &x, TVector &gradv, const bool only_elastic);

		void smoothing(const TVector &x, TVector &new_x);
		bool remesh(TVector &x);

		bool is_step_valid(const TVector &x0, const TVector &x1) const;
		bool is_intersection_free(const TVector &x) const;
		bool is_step_collision_free(const TVector &x0, const TVector &x1) const;
		double max_step_size(const TVector &x0, const TVector &x1) const;

		void line_search_begin(const TVector &x0, const TVector &x1);
		void line_search_end(bool failed);
		void post_step(const int iter_num, const TVector &x);
		void save_to_file(const TVector &x0);

		void solution_changed(const TVector &new_x);

		TVector get_lower_bound(const TVector &x) const;
		TVector get_upper_bound(const TVector &x) const;

		int n_inequality_constraints();
		TVector force_inequality_constraint(const TVector &x0, const TVector &dx);
		double inequality_constraint_val(const TVector &x, const int index);
		TVector inequality_constraint_grad(const TVector &x, const int index);

	private:
		std::vector<std::shared_ptr<Objective>> objectives_;
	};
} // namespace polyfem::solver

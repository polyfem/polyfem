#pragma once

#include <polyfem/solver/forms/Form.hpp>

#include <cppoptlib/problem.h>

#include <memory>
#include <vector>

namespace polyfem::solver
{
	class FullNLProblem : public cppoptlib::Problem<double>
	{
	public:
		using typename cppoptlib::Problem<double>::Scalar;
		using typename cppoptlib::Problem<double>::TVector;
		typedef StiffnessMatrix THessian;

		FullNLProblem(std::vector<std::shared_ptr<Form>> &forms);
		virtual void init(const TVector &x0);

		virtual double value(const TVector &x) override;
		double target_value(const TVector &x) { return value(x); }
		virtual void gradient(const TVector &x, TVector &gradv) override;
		void target_gradient(const TVector &x, TVector &gradv) { gradient(x, gradv); }
		virtual void hessian(const TVector &x, THessian &hessian);

		virtual bool is_step_valid(const TVector &x0, const TVector &x1);
		virtual bool is_step_collision_free(const TVector &x0, const TVector &x1);
		virtual double max_step_size(const TVector &x0, const TVector &x1);

		virtual void line_search_begin(const TVector &x0, const TVector &x1);
		virtual void line_search_end(bool failed);
		virtual void post_step(const int iter_num, const TVector &x);

		virtual void set_project_to_psd(bool val);

		virtual void solution_changed(const TVector &newX);

		virtual void init_lagging(const TVector &x);
		virtual void update_lagging(const TVector &x);

		TVector force_inequality_constraint(const TVector &x0, const TVector &dx) { return x0 + dx; }
		int n_inequality_constraints() { return 0; }
		double inequality_constraint_val(const TVector &x, const int index)
		{
			assert(false);
			return std::nan("");
		}
		TVector inequality_constraint_grad(const TVector &x, const int index)
		{
			assert(false);
			return TVector();
		}

		bool remesh(TVector &x) { return false; }
		void smoothing(const TVector &x, TVector &new_x) {}
		void save_to_file(const TVector &x0) {}

	protected:
		std::vector<std::shared_ptr<Form>> forms_;
	};
} // namespace polyfem::solver

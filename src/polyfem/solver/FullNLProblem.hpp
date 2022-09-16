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
		virtual void gradient(const TVector &x, TVector &gradv) override;
		virtual void hessian(const TVector &x, THessian &hessian);

		virtual bool is_step_valid(const TVector &x0, const TVector &x1) const;
		virtual bool is_step_collision_free(const TVector &x0, const TVector &x1) const;
		virtual double max_step_size(const TVector &x0, const TVector &x1) const;

		virtual void line_search_begin(const TVector &x0, const TVector &x1);
		virtual void line_search_end();
		virtual void post_step(const int iter_num, const TVector &x);

		virtual void set_project_to_psd(bool val);

		virtual void solution_changed(const TVector &new_x);

		virtual void init_lagging(const TVector &x);
		virtual bool update_lagging(const TVector &x, const int iter_num);
		bool uses_lagging() const;

	protected:
		std::vector<std::shared_ptr<Form>> forms_;
	};
} // namespace polyfem::solver

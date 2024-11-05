#include "FullNLProblem.hpp"
#include <ipc/utils/eigen_ext.hpp>


namespace polyfem::solver
{
	FullNLProblem::FullNLProblem(const std::vector<std::shared_ptr<Form>> &forms)
		: forms_(forms)
	{
	}

	void FullNLProblem::init(const TVector &x)
	{
		for (auto &f : forms_)
			f->init(x);
	}

	void FullNLProblem::set_project_to_psd(ipc::PSDProjectionMethod project_to_psd)
	{
		for (auto &f : forms_)
			f->set_project_to_psd(project_to_psd);
	}

	void FullNLProblem::init_lagging(const TVector &x)
	{
		for (auto &f : forms_)
			f->init_lagging(x);
	}

	void FullNLProblem::update_lagging(const TVector &x, const int iter_num)
	{
		for (auto &f : forms_)
			f->update_lagging(x, iter_num);
	}

	int FullNLProblem::max_lagging_iterations() const
	{
		int max_lagging_iterations = 1;
		for (auto &f : forms_)
			max_lagging_iterations = std::max(max_lagging_iterations, f->max_lagging_iterations());
		return max_lagging_iterations;
	}

	bool FullNLProblem::uses_lagging() const
	{
		for (auto &f : forms_)
			if (f->uses_lagging())
				return true;
		return false;
	}

	void FullNLProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		for (auto &f : forms_)
			f->line_search_begin(x0, x1);
	}

	void FullNLProblem::line_search_end()
	{
		for (auto &f : forms_)
			f->line_search_end();
	}

	double FullNLProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		double step = 1;
		for (auto &f : forms_)
			if (f->enabled())
				step = std::min(step, f->max_step_size(x0, x1));
		return step;
	}

	bool FullNLProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		for (auto &f : forms_)
			if (f->enabled() && !f->is_step_valid(x0, x1))
				return false;
		return true;
	}

	bool FullNLProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
	{
		for (auto &f : forms_)
			if (f->enabled() && !f->is_step_collision_free(x0, x1))
				return false;
		return true;
	}

	double FullNLProblem::value(const TVector &x)
	{
		double val = 0;
		for (auto &f : forms_)
			if (f->enabled())
				val += f->value(x);
		return val;
	}

	void FullNLProblem::gradient(const TVector &x, TVector &grad)
	{
		grad = TVector::Zero(x.size());
		for (auto &f : forms_)
		{
			if (!f->enabled())
				continue;
			TVector tmp;
			f->first_derivative(x, tmp);
			grad += tmp;
		}
	}

	void FullNLProblem::hessian(const TVector &x, THessian &hessian)
	{
		hessian.resize(x.size(), x.size());
		for (auto &f : forms_)
		{
			if (!f->enabled())
				continue;
			THessian tmp;
			f->second_derivative(x, tmp);
			hessian += tmp;
		}
	}

	void FullNLProblem::solution_changed(const TVector &x)
	{
		for (auto &f : forms_)
			f->solution_changed(x);
	}

	void FullNLProblem::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		for (auto &f : forms_)
			f->post_step(data);
	}
} // namespace polyfem::solver

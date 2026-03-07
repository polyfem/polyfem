#include "CompositeForm.hpp"
#include <polyfem/State.hpp>

namespace polyfem::solver
{
	Eigen::MatrixXd CompositeForm::compute_reduced_adjoint_rhs(const Eigen::VectorXd &x, const State &state) const
	{
		Eigen::VectorXd composite_grad = compose_grad(get_inputs(x));

		Eigen::MatrixXd term;
		Eigen::VectorXd tmp_grad;
		for (int i = 0; i < forms_.size(); i++)
		{
			if (i == 0)
				term = composite_grad(i) * forms_[i]->compute_reduced_adjoint_rhs(x, state);
			else
				term += composite_grad(i) * forms_[i]->compute_reduced_adjoint_rhs(x, state);
		}

		return term * weight();
	}

	void CompositeForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		Eigen::VectorXd composite_grad = compose_grad(get_inputs(x));

		gradv.setZero(x.size());
		Eigen::VectorXd tmp_grad;
		for (int i = 0; i < forms_.size(); i++)
		{
			forms_[i]->compute_partial_gradient(x, tmp_grad);
			gradv += composite_grad(i) * tmp_grad;
		}
		gradv *= weight();
	}

	Eigen::VectorXd CompositeForm::get_inputs(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd values;
		values.setZero(forms_.size());

		for (int i = 0; i < forms_.size(); i++)
			values(i) = forms_[i]->value(x);

		return values;
	}

	double CompositeForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return compose(get_inputs(x));
	}

	void CompositeForm::init(const Eigen::VectorXd &x)
	{
		for (const auto &f : forms_)
			f->init(x);
	}

	bool CompositeForm::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		for (const auto &f : forms_)
		{
			if (f->enabled() && !f->is_step_valid(x0, x1))
				return false;
		}
		return true;
	}

	double CompositeForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		double step = 1;
		for (const auto &f : forms_)
			if (f->enabled())
				step = std::min(step, f->max_step_size(x0, x1));

		return step;
	}

	void CompositeForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		for (const auto &f : forms_)
			f->line_search_begin(x0, x1);
	}

	void CompositeForm::line_search_end()
	{
		for (const auto &f : forms_)
			f->line_search_end();
	}

	void CompositeForm::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		for (const auto &f : forms_)
			f->post_step(data);
	}

	void CompositeForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		AdjointForm::solution_changed(new_x);
		for (const auto &f : forms_)
			f->solution_changed(new_x);
	}
	bool CompositeForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		for (const auto &f : forms_)
		{
			if (f->enabled() && !f->is_step_collision_free(x0, x1))
				return false;
		}
		return true;
	}
} // namespace polyfem::solver

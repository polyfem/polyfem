#include "TransientForm.hpp"
#include <polyfem/State.hpp>
#include <polyfem/io/MatrixIO.hpp>

namespace polyfem::solver
{
	std::vector<double> TransientForm::get_transient_quadrature_weights() const
	{
		std::vector<double> weights;
		weights.assign(time_steps_ + 1, dt_);
		if (transient_integral_type_ == "uniform")
		{
			weights[0] = 0;
		}
		else if (transient_integral_type_ == "trapezoidal")
		{
			weights[0] = dt_ / 2.;
			weights[weights.size() - 1] = dt_ / 2.;
		}
		else if (transient_integral_type_ == "simpson")
		{
			weights[0] = dt_ / 3.;
			weights[weights.size() - 1] = dt_ / 3.;
			for (int i = 1; i < weights.size() - 1; i++)
			{
				if (i % 2)
					weights[i] = dt_ * 4. / 3.;
				else
					weights[i] = dt_ * 2. / 4.;
			}
		}
		else if (transient_integral_type_ == "final")
		{
			weights.assign(time_steps_ + 1, 0);
			weights[time_steps_] = 1;
		}
		else if (transient_integral_type_ == "steps")
		{
			weights.assign(time_steps_ + 1, 0);
			for (const int step : steps_)
			{
				assert(step > 0 && step < weights.size());
				weights[step] += 1. / steps_.size();
			}
		}
		else
			assert(false);

		return weights;
	}

	double TransientForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		std::vector<double> weights = get_transient_quadrature_weights();

		double value = 0;
		for (int i = 0; i < time_steps_ + 1; i++)
		{
			if (weights[i] == 0)
				continue;
			const double tmp = obj_->value_unweighted_step(i, x);
			value += (weights[i] * obj_->weight()) * tmp;
		}

		return value;
	}
	Eigen::MatrixXd TransientForm::compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state) const
	{
		Eigen::MatrixXd terms;
		terms.setZero(state.ndof(), time_steps_ + 1);
		std::vector<double> weights = get_transient_quadrature_weights();

		for (int i = 0; i < time_steps_ + 1; i++)
		{
			if (weights[i] == 0)
				continue;
			terms.col(i) = weights[i] * obj_->compute_adjoint_rhs_step(i, x, state);
			if (obj_->depends_on_step_prev() && i > 0)
				terms.col(i - 1) = weights[i] * obj_->compute_adjoint_rhs_step_prev(i, x, state);
		}

		return terms * weight();
	}
	void TransientForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv.setZero(x.size());
		std::vector<double> weights = get_transient_quadrature_weights();

		Eigen::VectorXd tmp;
		for (int i = 0; i < time_steps_ + 1; i++)
		{
			if (weights[i] == 0)
				continue;
			obj_->compute_partial_gradient_step(i, x, tmp);
			gradv += weights[i] * tmp;
		}

		gradv *= weight();
	}

	void TransientForm::init(const Eigen::VectorXd &x)
	{
		obj_->init(x);
	}

	bool TransientForm::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return obj_->is_step_valid(x0, x1);
	}

	double TransientForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return obj_->max_step_size(x0, x1);
	}

	void TransientForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		obj_->line_search_begin(x0, x1);
	}

	void TransientForm::line_search_end()
	{
		obj_->line_search_end();
	}

	void TransientForm::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		obj_->post_step(data);
	}

	void TransientForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		AdjointForm::solution_changed(new_x);
		// obj_->solution_changed(new_x);
		std::vector<double> weights = get_transient_quadrature_weights();
		for (int i = 0; i <= time_steps_; i++)
		{
			if (weights[i] == 0)
				continue;
			obj_->solution_changed_step(i, new_x);
		}
	}
	bool TransientForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return obj_->is_step_collision_free(x0, x1);
	}

	double ProxyTransientForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd vals(steps_.size());
		int j = 0;
		for (int i : steps_)
		{
			vals(j++) = obj_->value_unweighted_step(i, x);
		}

		return eval(vals);
	}
	Eigen::MatrixXd ProxyTransientForm::compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state) const
	{
		Eigen::VectorXd vals(steps_.size());
		Eigen::MatrixXd terms;
		terms.setZero(state.ndof(), steps_.size());

		int j = 0;
		for (int i : steps_)
		{
			vals(j) = obj_->value_unweighted_step(i, x);
			terms.col(j++) = obj_->compute_adjoint_rhs_step(i, x, state);
		}

		const Eigen::VectorXd g = eval_grad(vals);
		Eigen::MatrixXd out;
		out.setZero(state.ndof(), time_steps_ + 1);
		j = 0;
		for (int i : steps_)
		{
			out.col(i) = terms.col(j) * (g(j) * weight());
			j++;
		}

		return out;
	}
	void ProxyTransientForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		Eigen::VectorXd vals(steps_.size());
		Eigen::MatrixXd terms;
		terms.setZero(x.size(), steps_.size());

		int j = 0;
		Eigen::VectorXd tmp;
		for (int i : steps_)
		{
			vals(j) = obj_->value_unweighted_step(i, x);
			obj_->compute_partial_gradient_step(i, x, tmp);
			terms.col(j++) = tmp;
		}

		gradv = terms * eval_grad(vals) * weight();
	}

	double ProxyTransientForm::eval(const Eigen::VectorXd &y) const
	{
		return 1. / y.array().inverse().sum();
	}

	Eigen::VectorXd ProxyTransientForm::eval_grad(const Eigen::VectorXd &y) const
	{
		return y.array().pow(-2.0) * pow(eval(y), 2);
	}
} // namespace polyfem::solver
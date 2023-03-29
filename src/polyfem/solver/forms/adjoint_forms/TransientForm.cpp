#include "TransientForm.hpp"
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
				weights[step] = 1. / steps_.size();
			}
		}
		else
			assert(false);

		return weights;
	}

	double TransientForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		double value = 0;
		std::vector<double> weights = get_transient_quadrature_weights();
		for (int i = 0; i <= time_steps_; i++)
		{
			if (weights[i] == 0)
				continue;
			obj_->set_time_step(i);
			const double tmp = obj_->value(x);
			value += weights[i] * tmp;
		}
		return value;
	}
	Eigen::MatrixXd TransientForm::compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state)
	{
		Eigen::MatrixXd terms;
		terms.setZero(state.ndof(), time_steps_ + 1);

		std::vector<double> weights = get_transient_quadrature_weights();
		for (int i = 0; i <= time_steps_; i++)
		{
			if (weights[i] == 0)
				continue;
			obj_->set_time_step(i);
			terms.col(i) = weights[i] * obj_->compute_adjoint_rhs_unweighted_step(x, state) * obj_->weight();
		}

		return terms;
	}
	void TransientForm::compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv.setZero(x.size());

		std::vector<double> weights = get_transient_quadrature_weights();
		Eigen::VectorXd tmp;
		for (int i = 0; i <= time_steps_; i++)
		{
			if (weights[i] == 0)
				continue;
			obj_->set_time_step(i);
			obj_->compute_partial_gradient(x, tmp);
			gradv += weights[i] * tmp;
		}
	}

	void TransientForm::init(const Eigen::VectorXd &x)
	{
		for (int i = 0; i <= time_steps_; i++)
		{
			obj_->set_time_step(i);
			obj_->init(x);
		}
	}

	bool TransientForm::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		for (int i = 0; i <= time_steps_; i++)
		{
			obj_->set_time_step(i);
			if (!obj_->is_step_valid(x0, x1))
				return false;
		}
		return true;
	}

	double TransientForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		double step = 1;
		for (int i = 0; i <= time_steps_; i++)
		{
			obj_->set_time_step(i);
			step = std::min(step, obj_->max_step_size(x0, x1));
		}

		return step;
	}

	void TransientForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		for (int i = 0; i <= time_steps_; i++)
		{
			obj_->set_time_step(i);
			obj_->line_search_begin(x0, x1);
		}
	}

	void TransientForm::line_search_end()
	{
		for (int i = 0; i <= time_steps_; i++)
		{
			obj_->set_time_step(i);
			obj_->line_search_end();
		}
	}

	void TransientForm::post_step(const int iter_num, const Eigen::VectorXd &x)
	{
		for (int i = 0; i <= time_steps_; i++)
		{
			obj_->set_time_step(i);
			obj_->post_step(iter_num, x);
		}
	}

	void TransientForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		for (int i = 0; i <= time_steps_; i++)
		{
			obj_->set_time_step(i);
			obj_->solution_changed(new_x);
		}
	}

	void TransientForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		for (int i = 0; i <= time_steps_; i++)
		{
			obj_->set_time_step(i);
			obj_->update_quantities(t, x);
		}
	}

	void TransientForm::init_lagging(const Eigen::VectorXd &x)
	{
		for (int i = 0; i <= time_steps_; i++)
		{
			obj_->set_time_step(i);
			obj_->init_lagging(x);
		}
	}

	void TransientForm::update_lagging(const Eigen::VectorXd &x, const int iter_num)
	{
		for (int i = 0; i <= time_steps_; i++)
		{
			obj_->set_time_step(i);
			obj_->update_lagging(x, iter_num);
		}
	}

	void TransientForm::set_apply_DBC(const Eigen::VectorXd &x, bool apply_DBC)
	{
		for (int i = 0; i <= time_steps_; i++)
		{
			obj_->set_time_step(i);
			obj_->set_apply_DBC(x, apply_DBC);
		}
	}

	bool TransientForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		for (int i = 0; i <= time_steps_; i++)
		{
			obj_->set_time_step(i);
			if (!obj_->is_step_collision_free(x0, x1))
				return false;
		}
		return true;
	}
} // namespace polyfem::solver
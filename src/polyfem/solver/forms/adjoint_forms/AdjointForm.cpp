#include "AdjointForm.hpp"

namespace polyfem::solver
{
    void AdjointForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        gradv.setZero(x.size());
        for (const auto &param_map : variable_to_simulations_)
        {
            const auto &parametrization = param_map->get_parameterization();
            const auto &state = param_map->get_state();
            const auto &param_type = param_map->get_parameter_type();

            gradv += parametrization.apply_jacobian(compute_adjoint_term(state, param_type), x);
        }

        Eigen::VectorXd partial_grad;
        compute_partial_gradient(x, partial_grad);
        gradv += partial_grad;
    }

    void AdjointForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        log_and_throw_error("Should override this function in any AdjointForm!");
    }

    Eigen::MatrixXd AdjointForm::compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state)
    {
        log_and_throw_error("Should override this function in any AdjointForm!");
        return Eigen::MatrixXd();
    }

    Eigen::VectorXd AdjointForm::compute_adjoint_term(const State &state, const ParameterType &param) const
    {
        Eigen::VectorXd term;
        AdjointTools::compute_adjoint_term(state, state.get_adjoint_mat(), param, term);
        return term;
    }

    Eigen::MatrixXd StaticForm::compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state)
    {
		Eigen::MatrixXd term(state.ndof(), state.diff_cached.size());
		term.col(time_step_) = compute_adjoint_rhs_step(x, state);

		return term;
    }

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
		else if (transient_integral_type_.find("step_") != std::string::npos)
		{
			weights.assign(time_steps_ + 1, 0);
			int step = std::stoi(transient_integral_type_.substr(5));
			assert(step > 0 && step < weights.size());
			weights[step] = 1;
		}
		else if (json::parse(transient_integral_type_).is_array())
		{
			weights.assign(time_steps_ + 1, 0);
			auto steps = json::parse(transient_integral_type_);
			for (const int step : steps)
			{
				assert(step > 0 && step < weights.size());
				weights[step] = 1. / steps.size();
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
			value += weights[i] * obj_->value_unweighted(x);
		}
		return value;
    }
    Eigen::MatrixXd TransientForm::compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state)
    {
		Eigen::MatrixXd terms;
		terms.setZero(state.ndof(), time_steps_ + 1);

		std::vector<double> weights = get_transient_quadrature_weights();
		for (int i = 0; i <= time_steps_; i++)
		{
			if (weights[i] == 0)
				continue;
			obj_->set_time_step(i);
			terms.col(i) = weights[i] * obj_->compute_adjoint_rhs_step(x, state);
		}

		return terms;
    }
    void TransientForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
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

}
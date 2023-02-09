#pragma once

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	class ParameterizationForm : public Form
	{
	public:
		virtual ~ParameterizationForm() {}

		virtual void init(const Eigen::VectorXd &x) final override
		{
			init_with_param(apply_parameterizations(x));
		}

		inline virtual double value(const Eigen::VectorXd &x) const final override
		{
			return weight_ * value_unweighted(apply_parameterizations(x));
		}

		virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
		{
			Eigen::VectorXd y = apply_parameterizations(x);
			first_derivative_unweighted_with_param(y, gradv);

			gradv = apply_jacobian(x, gradv);
		}

		inline void second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const final override
		{
			log_and_throw("Not implemented");
			second_derivative_unweighted(x, hessian);
			hessian *= weight_;
		}

		virtual bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const final override
		{
			return is_step_valid_with_param(apply_parameterizations(x0), apply_parameterizations(x1));
		}

		virtual double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const final
		{
			return max_step_size_with_param(apply_parameterizations(x0), apply_parameterizations(x1));
		}

		virtual void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) final override
		{
			line_search_begin_with_param(apply_parameterizations(x0), apply_parameterizations(x1));
		}

		virtual void post_step(const int iter_num, const Eigen::VectorXd &x) final override
		{
			post_step_with_param(iter_num, apply_parameterizations(x));
		}

		virtual void solution_changed(const Eigen::VectorXd &new_x) final override
		{
			solution_changed_with_param(apply_parameterizations(new_x));
		}

		virtual void update_quantities(const double t, const Eigen::VectorXd &x) final override
		{
			update_quantities_with_param(t, apply_parameterizations(x));
		}

		virtual void init_lagging(const Eigen::VectorXd &x) final override
		{
			init_lagging_with_param(apply_parameterizations(x));
		}

		virtual void update_lagging(const Eigen::VectorXd &x, const int iter_num) final override
		{
			update_lagging_with_param(apply_parameterizations(x), iter_num);
		}

		virtual void set_apply_DBC(const Eigen::VectorXd &x, bool apply_DBC) final override
		{
			set_apply_DBC_with_param(apply_parameterizations(x), apply_DBC);
		}

		virtual bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const final override
		{
			return is_step_collision_free_with_param(apply_parameterizations(x0), apply_parameterizations(x1));
		}

	protected:
		virtual void init_with_param(const Eigen::VectorXd &x) {}
		virtual bool is_step_valid_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) { return true }
		virtual double max_step_size_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) { return 1; }
		virtual void line_search_begin_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) {}
		virtual void post_step_with_param(const int iter_num, const Eigen::VectorXd &x) {}
		virtual void solution_changed_with_param(const Eigen::VectorXd &new_x) {}
		virtual void update_quantities_with_param(const double t, const Eigen::VectorXd &x) {}
		virtual void init_lagging_with_param(const Eigen::VectorXd &x) {}
		virtual void update_lagging_with_param(const Eigen::VectorXd &x, const int iter_num) {}
		virtual void set_apply_DBC_with_param(const Eigen::VectorXd &x, bool apply_DBC) {}
		virtual bool is_step_collision_free_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) { return true; }
		virtual void first_derivative_unweighted_with_param(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const {}

		virtual Eigen::VectorXd compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
		{
			return first_derivative_unweighted(x, gradv);
		}

	private:
		CompositeParameterization parameterizations_;

		Eigen::VectorXd apply_parameterizations(const Eigen::VectorXd &x) const
		{
			return parameterizations_.eval(x);
		}
	};
} // namespace polyfem::solver

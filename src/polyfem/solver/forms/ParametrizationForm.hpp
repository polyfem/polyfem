#pragma once

#include <polyfem/utils/Types.hpp>
#include <polyfem/solver/forms/parametrization/Parametrization.hpp>
#include "Form.hpp"
#include <polyfem/State.hpp>

namespace polyfem::solver
{
	class ParametrizationForm : public Form
	{
	public:
		ParametrizationForm(const CompositeParametrization &parametrizations) : parametrizations_(parametrizations) {}
		virtual ~ParametrizationForm() {}

		virtual void init(const Eigen::VectorXd &x) final override
		{
			init_with_param(apply_parametrizations(x));
		}

		virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
		{
			Eigen::VectorXd y = apply_parametrizations(x);
			first_derivative_unweighted_with_param(y, gradv);

			gradv = parametrizations_.apply_jacobian(gradv, x);
		}

		inline virtual void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const final override
		{
			log_and_throw_error("Not implemented");
		}

		virtual bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const final override
		{
			return is_step_valid_with_param(apply_parametrizations(x0), apply_parametrizations(x1));
		}

		virtual double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const final
		{
			return max_step_size_with_param(apply_parametrizations(x0), apply_parametrizations(x1));
		}

		virtual void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) final override
		{
			line_search_begin_with_param(apply_parametrizations(x0), apply_parametrizations(x1));
		}

		virtual void line_search_end() final override
		{
			line_search_end_with_param();
		}

		virtual void post_step(const int iter_num, const Eigen::VectorXd &x) final override
		{
			post_step_with_param(iter_num, apply_parametrizations(x));
		}

		virtual void solution_changed(const Eigen::VectorXd &new_x) final override
		{
			solution_changed_with_param(apply_parametrizations(new_x));
		}

		virtual void update_quantities(const double t, const Eigen::VectorXd &x) final override
		{
			update_quantities_with_param(t, apply_parametrizations(x));
		}

		virtual void init_lagging(const Eigen::VectorXd &x) final override
		{
			init_lagging_with_param(apply_parametrizations(x));
		}

		virtual void update_lagging(const Eigen::VectorXd &x, const int iter_num) final override
		{
			update_lagging_with_param(apply_parametrizations(x), iter_num);
		}

		virtual void set_apply_DBC(const Eigen::VectorXd &x, bool apply_DBC) final override
		{
			set_apply_DBC_with_param(apply_parametrizations(x), apply_DBC);
		}

		virtual bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const final override
		{
			return is_step_collision_free_with_param(apply_parametrizations(x0), apply_parametrizations(x1));
		}

		virtual Eigen::MatrixXd compute_adjoint_rhs(const Eigen::VectorXd &x, const State &state) { return Eigen::MatrixXd::Zero(state.ndof(), state.diff_cached.size()); }

	protected:
		virtual void init_with_param(const Eigen::VectorXd &x) {}
		virtual bool is_step_valid_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const { return true; }
		virtual double max_step_size_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const { return 1; }
		virtual void line_search_begin_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) {}
		virtual void line_search_end_with_param() {}
		virtual void post_step_with_param(const int iter_num, const Eigen::VectorXd &x) {}
		virtual void solution_changed_with_param(const Eigen::VectorXd &new_x) {}
		virtual void update_quantities_with_param(const double t, const Eigen::VectorXd &x) {}
		virtual void init_lagging_with_param(const Eigen::VectorXd &x) {}
		virtual void update_lagging_with_param(const Eigen::VectorXd &x, const int iter_num) {}
		virtual void set_apply_DBC_with_param(const Eigen::VectorXd &x, bool apply_DBC) {}
		virtual bool is_step_collision_free_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const { return true; }
		virtual void first_derivative_unweighted_with_param(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const {}

		virtual void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
		{
			first_derivative_unweighted(x, gradv);
		}

	private:
		CompositeParametrization parametrizations_;

		Eigen::VectorXd apply_parametrizations(const Eigen::VectorXd &x) const
		{
			return parametrizations_.eval(x);
		}
	};
} // namespace polyfem::solver

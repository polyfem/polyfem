#pragma once

#include <polyfem/utils/Types.hpp>
#include <polyfem/optimization/parametrization/Parametrization.hpp>
#include <polyfem/optimization/forms/AdjointForm.hpp>

namespace polyfem
{
	class State;
}

namespace polyfem::solver
{
	class ParametrizationForm : public AdjointForm
	{
	public:
		ParametrizationForm(CompositeParametrization &&parametrizations) : AdjointForm({}), parametrizations_(parametrizations) {}
		virtual ~ParametrizationForm() {}

		virtual void init(const Eigen::VectorXd &x) final override;
		virtual bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const final override;
		virtual double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const final;
		virtual void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) final override;
		virtual void line_search_end() final override;
		virtual void post_step(const polysolve::nonlinear::PostStepData &data) final override;
		virtual void solution_changed(const Eigen::VectorXd &new_x) final override;
		virtual bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const final override;

	protected:
		virtual double value_unweighted(const Eigen::VectorXd &x) const final override;
		virtual void compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const final override;

		virtual void init_with_param(const Eigen::VectorXd &x) {}
		virtual bool is_step_valid_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const { return true; }
		virtual double max_step_size_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const { return 1; }
		virtual void line_search_begin_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) {}
		virtual void line_search_end_with_param() {}
		virtual void post_step_with_param(const polysolve::nonlinear::PostStepData &data) {}
		virtual void solution_changed_with_param(const Eigen::VectorXd &new_x) {}
		virtual bool is_step_collision_free_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const { return true; }
		virtual double value_unweighted_with_param(const Eigen::VectorXd &x) const { return 0; }
		virtual void compute_partial_gradient_with_param(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const = 0;

	private:
		CompositeParametrization parametrizations_;

		inline Eigen::VectorXd apply_parametrizations(const Eigen::VectorXd &x) const
		{
			return parametrizations_.eval(x);
		}
	};
} // namespace polyfem::solver

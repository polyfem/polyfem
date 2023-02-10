#pragma once

#include "AdjointForm.hpp"
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	class CompositeForm : public AdjointForm
	{
	public:
		CompositeForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations, const std::vector<std::shared_ptr<ParametrizationForm>> &forms) : AdjointForm(variable_to_simulations, parametrizations), forms_(forms) {}
		CompositeForm(const std::vector<std::shared_ptr<ParametrizationForm>> &forms) : AdjointForm({}, CompositeParametrization()), forms_(forms) {}
		virtual ~CompositeForm() {}

		virtual int n_objs() const final { return forms_.size(); }

		virtual Eigen::MatrixXd compute_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state)
		{
			Eigen::VectorXd composite_grad = compose_grad(get_inputs(x));

			Eigen::MatrixXd term;
			term.setZero(state.ndof(), state.diff_cached.size());
			Eigen::VectorXd tmp_grad;
			for (int i = 0; i < forms_.size(); i++)
				term += composite_grad(i) * forms_[i]->compute_adjoint_rhs(x, state); // important: not "unweighted"

			return term;
		}

		virtual void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
		{
			Eigen::VectorXd composite_grad = compose_grad(get_inputs(x));

			gradv.setZero(x.size());
			Eigen::VectorXd tmp_grad;
			for (int i = 0; i < forms_.size(); i++)
			{
				forms_[i]->compute_partial_gradient(x, tmp_grad); // important: not "unweighted"
				gradv += composite_grad(i) * tmp_grad;
			}
		}

	protected:
		virtual double compose(const Eigen::VectorXd &inputs) const = 0;
		virtual Eigen::VectorXd compose_grad(const Eigen::VectorXd &inputs) const = 0;

		Eigen::VectorXd get_inputs(const Eigen::VectorXd &x) const
		{
			Eigen::VectorXd values;
			values.setZero(forms_.size());

			for (int i = 0; i < forms_.size(); i++)
				values(i) = forms_[i]->value(x);

			return values;
		}

		double value_unweighted(const Eigen::VectorXd &x) const override
		{
			return compose(get_inputs(x));
		}

		virtual void init_with_param(const Eigen::VectorXd &x) override
		{
			for (const auto &f : forms_)
				f->init_with_param(x);
		}

		virtual bool is_step_valid_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			for (const auto &f : forms_)
			{
				if (f->enabled() && !f->is_step_valid_with_param(x0, x1))
					return false;
			}
			return true;
		}

		virtual double max_step_size_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			double step = 1;
			for (const auto &f : forms_)
				if (f->enabled())
					step = std::min(step, f->max_step_size_with_param(x0, x1));

			return step;
		}

		virtual void line_search_begin_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override
		{
			for (const auto &f : forms_)
				f->line_search_begin_with_param(x0, x1);
		}

		virtual void line_search_end_with_param() override
		{
			for (const auto &f : forms_)
				f->line_search_end_with_param();
		}

		virtual void post_step_with_param(const int iter_num, const Eigen::VectorXd &x) override
		{
			for (const auto &f : forms_)
				f->post_step_with_param(iter_num, x);
		}

		virtual void solution_changed_with_param(const Eigen::VectorXd &new_x) override
		{
			for (const auto &f : forms_)
				f->solution_changed_with_param(new_x);
		}

		virtual void update_quantities_with_param(const double t, const Eigen::VectorXd &x) override
		{
			for (const auto &f : forms_)
				f->update_quantities_with_param(t, x);
		}

		virtual void init_lagging_with_param(const Eigen::VectorXd &x) override
		{
			for (const auto &f : forms_)
				f->init_lagging_with_param(x);
		}

		virtual void update_lagging_with_param(const Eigen::VectorXd &x, const int iter_num) override
		{
			for (const auto &f : forms_)
				f->update_lagging_with_param(x, iter_num);
		}

		virtual void set_apply_DBC_with_param(const Eigen::VectorXd &x, bool apply_DBC) override
		{
			for (const auto &f : forms_)
				f->set_apply_DBC_with_param(x, apply_DBC);
		}

		virtual bool is_step_collision_free_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			for (const auto &f : forms_)
			{
				if (f->enabled() && !f->is_step_collision_free_with_param(x0, x1))
					return false;
			}
			return true;
		}

		// TODO: should be user input
		int max_lagging_iterations() const override
		{
			int max_lagging_iterations = 1;
			for (const auto &f : forms_)
				max_lagging_iterations = std::max(max_lagging_iterations, f->max_lagging_iterations());
			return max_lagging_iterations;
		}

		// TODO: should be user input
		bool uses_lagging() const override
		{
			for (const auto &f : forms_)
				if (f->uses_lagging())
					return true;
			return false;
		}

		// TODO: should be user input
		void set_project_to_psd(bool project_to_psd)
		{
			for (const auto &f : forms_)
				f->set_project_to_psd(project_to_psd);
		}

	private:
		std::vector<std::shared_ptr<ParametrizationForm>> forms_;
	};
} // namespace polyfem::solver

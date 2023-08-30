#pragma once

#include "AdjointForm.hpp"
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	class CompositeForm : public AdjointForm
	{
	public:
		CompositeForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const std::vector<std::shared_ptr<AdjointForm>> &forms) : AdjointForm(variable_to_simulations), forms_(forms) {}
		CompositeForm(const std::vector<std::shared_ptr<AdjointForm>> &forms) : AdjointForm(forms[0]->get_variable_to_simulations()), forms_(forms) {}
		virtual ~CompositeForm() {}

		virtual int n_objs() const final { return forms_.size(); }

		virtual Eigen::MatrixXd compute_reduced_adjoint_rhs_unweighted(const Eigen::VectorXd &x, const State &state) const override final
		{
			Eigen::VectorXd composite_grad = compose_grad(get_inputs(x));

			Eigen::MatrixXd term;
			Eigen::VectorXd tmp_grad;
			for (int i = 0; i < forms_.size(); i++)
			{
				if (i == 0)
					term = composite_grad(i) * forms_[i]->compute_adjoint_rhs(x, state);
				else
					term += composite_grad(i) * forms_[i]->compute_adjoint_rhs(x, state); // important: not "unweighted"
			}

			return term;
		}

		virtual void compute_partial_gradient_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override final
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

		virtual double value_unweighted(const Eigen::VectorXd &x) const final override
		{
			return compose(get_inputs(x));
		}

		virtual void init(const Eigen::VectorXd &x) override
		{
			for (const auto &f : forms_)
				f->init(x);
		}

		virtual bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			for (const auto &f : forms_)
			{
				if (f->enabled() && !f->is_step_valid(x0, x1))
					return false;
			}
			return true;
		}

		virtual double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			double step = 1;
			for (const auto &f : forms_)
				if (f->enabled())
					step = std::min(step, f->max_step_size(x0, x1));

			return step;
		}

		virtual void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override
		{
			for (const auto &f : forms_)
				f->line_search_begin(x0, x1);
		}

		virtual void line_search_end() override
		{
			for (const auto &f : forms_)
				f->line_search_end();
		}

		virtual void post_step(const int iter_num, const Eigen::VectorXd &x) override
		{
			for (const auto &f : forms_)
				f->post_step(iter_num, x);
		}

		virtual void solution_changed(const Eigen::VectorXd &new_x) override
		{
			AdjointForm::solution_changed(new_x);
			for (const auto &f : forms_)
				f->solution_changed(new_x);
		}

		virtual void update_quantities(const double t, const Eigen::VectorXd &x) override
		{
			for (const auto &f : forms_)
				f->update_quantities(t, x);
		}

		virtual void init_lagging(const Eigen::VectorXd &x) override
		{
			for (const auto &f : forms_)
				f->init_lagging(x);
		}

		virtual void update_lagging(const Eigen::VectorXd &x, const int iter_num) override
		{
			for (const auto &f : forms_)
				f->update_lagging(x, iter_num);
		}

		virtual void set_apply_DBC(const Eigen::VectorXd &x, bool apply_DBC) override
		{
			for (const auto &f : forms_)
				f->set_apply_DBC(x, apply_DBC);
		}

		virtual bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override
		{
			for (const auto &f : forms_)
			{
				if (f->enabled() && !f->is_step_collision_free(x0, x1))
					return false;
			}
			return true;
		}

	private:
		std::vector<std::shared_ptr<AdjointForm>> forms_;
	};
} // namespace polyfem::solver

#pragma once

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	class SumForm : public ParametrizationForm
	{
	public:
		virtual ~SumForm() {}

	protected:
		virtual double value_unweighted(const Eigen::VectorXd &x) const override
		{
			double value = 0;
			for (const auto f : forms_)
			{
				if (!f->enabled())
					continue;
				value += f->value_unweighted(x);
			}
		}

		virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
		{
			grad = VectorXd::Zero(x.size());
			VectorXd tmp;

			for (auto &f : forms_)
			{
				if (!f->enabled())
					continue;
				f->first_derivative_unweighted(x, tmp);
				grad += tmp;
			}
		}

		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
		{
			hessian.resize(x.size(), x.size());

			StiffnessMatrix tmp;
			for (auto &f : forms_)
			{
				if (!f->enabled())
					continue;
				f->second_derivative_unweighted(x, tmp);
				hessian += tmp;
			}
		}

		virtual void init_with_param(const Eigen::VectorXd &x) override
		{
			for (const auto f : forms_)
				f->init_with_param(x);
		}

		virtual bool is_step_valid_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override
		{
			for (const auto f : forms_)
			{
				if (f->enabled() && !f->is_step_valid_with_param(x0, x1))
					return false;
			}
			return true;
		}

		virtual double max_step_size_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override
		{
			double step = 1;
			for (const auto f : forms_)
				if (f->enabled())
					step = std::min(step, f->max_step_size_with_param(x0, x1));

			return step;
		}

		virtual void line_search_begin_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override
		{
			for (const auto f : forms_)
				f->line_search_begin_with_param(x0, x1);
		}

		virtual void FullNLProblem::line_search_end() override
		{
			for (auto &f : forms_)
				f->line_search_end();
		}

		virtual void post_step_with_param(const int iter_num, const Eigen::VectorXd &x) override
		{
			for (const auto f : forms_)
				f->post_step_with_param(iter_num, x);
		}

		virtual void solution_changed_with_param(const Eigen::VectorXd &new_x) override
		{
			for (const auto f : forms_)
				f->solution_changed_with_param(new_x);
		}

		virtual void update_quantities_with_param(const double t, const Eigen::VectorXd &x) override
		{
			for (const auto f : forms_)
				f->update_quantities_with_param(t, x);
		}

		virtual void init_lagging_with_param(const Eigen::VectorXd &x) override
		{
			for (const auto f : forms_)
				f->init_lagging_with_param(x);
		}

		virtual void update_lagging_with_param(const Eigen::VectorXd &x, const int iter_num) override
		{
			for (const auto f : forms_)
				f->update_lagging_with_param(x, iter_num);
		}

		virtual void set_apply_DBC_with_param(const Eigen::VectorXd &x, bool apply_DBC) override
		{
			for (const auto f : forms_)
				f->set_apply_DBC_with_param(x, apply_DBC);
		}

		virtual bool is_step_collision_free_with_param(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override
		{
			for (const auto f : forms_)
			{
				if (f->enabled() && !f->is_step_collision_free_with_param(x0, x1))
					return false;

				return true;
			}
		}

		int max_lagging_iterations() const override
		{
			int max_lagging_iterations = 1;
			for (auto &f : forms_)
				max_lagging_iterations = std::max(max_lagging_iterations, f->max_lagging_iterations());
			return max_lagging_iterations;
		}

		bool uses_lagging() const override
		{
			for (auto &f : forms_)
				if (f->uses_lagging())
					return true;
			return false;
		}

		void set_project_to_psd(bool project_to_psd)
		{
			for (auto &f : forms_)
				f->set_project_to_psd(project_to_psd);
		}

	private:
		std::vector<std::shared_ptr<Form>> forms_;
	};
} // namespace polyfem::solver

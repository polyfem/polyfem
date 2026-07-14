#include "FullNLProblem.hpp"
#include <polyfem/utils/Logger.hpp>

#include <algorithm>

namespace polyfem::solver
{
	namespace
	{
		/// Build form enabled/disabled mask.
		std::vector<uint8_t> build_enabled_mask(const std::vector<std::shared_ptr<Form>> &forms)
		{
			std::vector<uint8_t> enabled;
			enabled.reserve(forms.size());
			for (const auto &form : forms)
				enabled.push_back(form->enabled() ? 1 : 0);
			return enabled;
		}
	} // namespace

	FullNLProblem::FullNLProblem(const std::vector<std::shared_ptr<Form>> &forms, ExecutionPolicy policy, const bool is_residual)
		: forms_(forms),
		  execution_policy_(policy),
		  is_residual_(is_residual)
	{
	}

	double FullNLProblem::normalize_forms()
	{
		double total_weight = 0;
		for (const auto &f : forms_)
			total_weight += f->weight();

		logger().debug("Normalizing forms with scale: {}", total_weight);

		for (auto &f : forms_)
			f->set_scale(total_weight);

		return total_weight;
	}

	void FullNLProblem::init(const TVector &x)
	{
		for (auto &f : forms_)
			f->init(x);
	}

	void FullNLProblem::set_project_to_psd(bool project_to_psd)
	{
		for (auto &f : forms_)
			f->set_project_to_psd(project_to_psd);
	}

	void FullNLProblem::init_lagging(const TVector &x)
	{
		for (auto &f : forms_)
			f->init_lagging(x);
	}

	void FullNLProblem::update_lagging(const TVector &x, const int iter_num)
	{
		for (auto &f : forms_)
			f->update_lagging(x, iter_num);
	}

	int FullNLProblem::max_lagging_iterations() const
	{
		int max_lagging_iterations = 1;
		for (auto &f : forms_)
			max_lagging_iterations = std::max(max_lagging_iterations, f->max_lagging_iterations());
		return max_lagging_iterations;
	}

	bool FullNLProblem::uses_lagging() const
	{
		for (auto &f : forms_)
			if (f->uses_lagging())
				return true;
		return false;
	}

	void FullNLProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		for (auto &f : forms_)
			f->line_search_begin(x0, x1);
	}

	void FullNLProblem::line_search_end()
	{
		for (auto &f : forms_)
			f->line_search_end();
	}

	double FullNLProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		double step = 1;
		for (auto &f : forms_)
			if (f->enabled())
				step = std::min(step, f->max_step_size(x0, x1));
		return step;
	}

	bool FullNLProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		for (auto &f : forms_)
			if (f->enabled() && !f->is_step_valid(x0, x1))
				return false;
		return true;
	}

	bool FullNLProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
	{
		for (auto &f : forms_)
			if (f->enabled() && !f->is_step_collision_free(x0, x1))
				return false;
		return true;
	}

	double FullNLProblem::value(const TVector &x)
	{
		double val = 0;
		for (auto &f : forms_)
			if (f->enabled())
				val += f->value_ng(x, execution_policy_);
		return val;
	}

	void FullNLProblem::gradient(const TVector &x, TVector &grad)
	{
		DualVector grad_dual(x.size());
		for (auto &f : forms_)
		{
			if (!f->enabled())
				continue;
			f->first_derivative_ng(x, grad_dual, execution_policy_);
		}
		grad = grad_dual.to_eigen(execution_policy_);
	}

	void FullNLProblem::hessian(const TVector &x, THessian &hessian)
	{
		std::vector<uint8_t> enabled_mask = build_enabled_mask(forms_);

		// Compute sparsity pattern if cache not exists or form is enabled/disabled.
		if (!hessian_bsr_ || form_enabled_mask_ != enabled_mask)
		{
			std::optional<BSRSparsityPattern> joined_pattern;

			for (auto &f : forms_)
			{
				if (!f->enabled())
				{
					continue;
				}

				auto pattern = f->hessian_sparsity_pattern_ng();
				if (!pattern)
				{
					continue;
				}
				assert(pattern->rows == pattern->cols);
				assert(pattern->rows == x.size());

				if (joined_pattern)
				{
					joined_pattern->join(*pattern);
				}
				else
				{
					joined_pattern = std::move(pattern);
				}
			}

			if (joined_pattern)
			{
				hessian_bsr_.emplace(*joined_pattern);
			}
			else
			{
				// No pattern available, construct dyanmic only bsr matrix.
				hessian_bsr_.emplace(x.size(), x.size());
			}

			form_enabled_mask_ = std::move(enabled_mask);
		}

		// Reset value ptr to zero.
		hessian_bsr_->reset(execution_policy_);
		for (auto &f : forms_)
		{
			if (!f->enabled())
				continue;
			f->second_derivative_ng(x, *hessian_bsr_, execution_policy_);
		}

		hessian = hessian_bsr_->to_stiffness_matrix(execution_policy_);
	}

	void FullNLProblem::solution_changed(const TVector &x)
	{
		for (auto &f : forms_)
			f->solution_changed(x);
	}

	void FullNLProblem::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		for (auto &f : forms_)
			f->post_step(data);
	}
} // namespace polyfem::solver

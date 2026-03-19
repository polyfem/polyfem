#include "FullNLProblem.hpp"
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{
	FullNLProblem::FullNLProblem(const std::vector<std::shared_ptr<Form>> &forms, const bool is_residual)
		: forms_(forms),
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
				val += f->value(x);
		return val;
	}

	void FullNLProblem::gradient(const TVector &x, TVector &grad)
	{
		grad = TVector::Zero(x.size());
		for (auto &f : forms_)
		{
			if (!f->enabled())
				continue;
			TVector tmp;
			f->first_derivative(x, tmp);
			grad += tmp;
		}
	}

	void FullNLProblem::hessian(const TVector &x, THessian &hessian)
	{
		// hessian.resize(x.size(), x.size());
		// for (auto &f : forms_)
		// {
		// 	if (!f->enabled())
		// 		continue;
		// 	THessian tmp;
		// 	f->second_derivative(x, tmp);
		// 	hessian += tmp;
		// }

		updateHessianSparsityPattern();
		evalHessian(x);
		utils::NewtonHessian2SparseMatrix(m_hessianSparsity, hessian);
		
	}

	void FullNLProblem::evalHessian(const TVector &x)
	{
		for (auto &f : forms_)
		{
			if (!f->enabled())
				continue;
			f->accumulateHessian(f->weight(), x, m_hessianSparsity);
		}
	}

	bool FullNLProblem::updateHessianSparsityPattern()
	{
		NewtonHessian dynamicSparsity;
        const bool force = m_fullSparsityRebuildNeeded; // Force a rebuild of the sparsity pattern every time for now since the logic is still being developed and tested. We can disable this once we're confident in the correctness and efficiency of the sparsity pattern update logic.
        if (force) {
            m_hessianSparsity = NewtonHessian();
            m_hessianSparsityStaticPart = NewtonHessian();
		}

        bool changed = force;
        bool staticOnly = true;

        for (size_t i = 0; i < forms_.size(); ++i) {
            if (!forms_[i]->enabled())
				continue;
			const auto &f = forms_[i];
        
            if (f->sparsityPatternIsStatic()) {
                // Only rebuild the "static" part when the terms might have been invalidated.
                if (force) { m_hessianSparsityStaticPart.mergeSparsityPattern(f->hessianSparsityPattern()); std::cout << "Building static term sparsity pattern" << std::endl; }
            }
            else {
                dynamicSparsity.mergeSparsityPattern(f->hessianSparsityPattern());
                changed = true;
				staticOnly = false;
            }
        }

        if (changed) {
            if (staticOnly) m_hessianSparsity = std::move(m_hessianSparsityStaticPart);
            else {
                if (!m_sparsityLRU && m_hessianSparsityStaticPart.H_ss && (m_hessianSparsityStaticPart.H_ss->nnz() > 0)) {
                    m_sparsityLRU = std::make_unique<SparsityLRU>(*(m_hessianSparsityStaticPart.H_ss));
                    m_hessianSparsity = std::move(m_hessianSparsityStaticPart.H_ss);
                }

                if (m_sparsityLRU) {
                    changed = m_sparsityLRU->update(*(dynamicSparsity.H_ss));
                    changed |= force; // Ensure the static part rebuild takes effect.
                    if (changed) {
                        if (!m_hessianSparsity.H_ss) throw std::logic_error("NewtonMultiobjectiveProblem::m_updateSparsityPattern: m_hessianSparsity not initialized"); // This should never happen since `m_sparsityLRU` is only created when the static part is nonempty...
                        m_hessianSparsity.H_ss->Ap = (*m_sparsityLRU)->Ap;
                        m_hessianSparsity.H_ss->Ai = (*m_sparsityLRU)->Ai;
                        m_hessianSparsity.H_ss->nz = (*m_sparsityLRU)->nz;
                    }
                }
                else {
                    // No static part: hessian is purely dynamic, build it directly.
                    m_hessianSparsity = m_hessianSparsityStaticPart;
                    m_hessianSparsity.mergeSparsityPattern(dynamicSparsity);
                }
            }

            m_hessianSparsity.finalize();
        }
        else if (m_sparsityLRU) {
            // Still notify the cache of the sparsity pattern update in case
            // it triggers a refactorization due to entry expiration.
            if (m_sparsityLRU->increaseAgeOfOldEntries()) {
                if (!m_hessianSparsity.H_ss) throw std::logic_error("NewtonMultiobjectiveProblem::m_updateSparsityPattern: m_hessianSparsity not initialized"); // This should never happen since `m_sparsityLRU` is only created when the static part is nonempty...
                m_hessianSparsity.H_ss->Ap = (*m_sparsityLRU)->Ap;
                m_hessianSparsity.H_ss->Ai = (*m_sparsityLRU)->Ai;
                m_hessianSparsity.H_ss->nz = (*m_sparsityLRU)->nz;

                m_hessianSparsity.finalize();

                changed = true;
            }
        }

        m_fullSparsityRebuildNeeded = false;


		return changed;

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

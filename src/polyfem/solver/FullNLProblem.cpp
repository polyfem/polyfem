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

	void FullNLProblem::hessian(const TVector &x, polysolve::Hessian &hessian)
	{
#if 0 // Original Eigen-based implementation
		auto &H = hessian.emplace<StiffnessMatrix>();

		H.resize(x.size(), x.size());
		for (auto &f : forms_) {
			if (!f->enabled())
				continue;
			THessian tmp;
			f->second_derivative(x, tmp);
			H += tmp;
		}
#else
		auto &H = hessian.emplace<polysolve::BCSCHessianWithFixedVars>().H;

		bool changed = updateHessianSparsityPattern();
		if (changed) ++m_sparsityPatternID;

		if (m_hessianSparsity != nullptr)
			H = m_hessianSparsity->clone(); // Accumulate into a fresh (value-free) copy of the sparsity pattern.
		StiffnessMatrix unacceleratedHessianContribs(x.size(), x.size());
		for (auto &f : forms_) {
			if (!f->enabled()) continue;
			if (f->usesFastSystemAssembler()) {
				f->accumulateHessian(f->weight() / f->scale(), x, *H);
				continue;
			}

			// std::cout << "Evaluating unaccelerated Hessian for form: " << f->name() << std::endl;
			THessian tmp;
			f->second_derivative(x, tmp);
			unacceleratedHessianContribs += tmp;
		}

		if ((unacceleratedHessianContribs.nonZeros() > 0) || (!H)) {
			if (H && !H->isSparsityOnly()) unacceleratedHessianContribs += hessian.as<StiffnessMatrix>();
			hessian.emplace<StiffnessMatrix>(std::move(unacceleratedHessianContribs));
			if (!m_alreadyWarnedAboutUnacceleratedPath) {
				logger().warn("Some forms do not support fast system assembly; Falling back to unaccelerated path.");
				m_alreadyWarnedAboutUnacceleratedPath = true;
			}
			++m_sparsityPatternID; // Sparsity pattern likely changed (all bets are off)...
		}
#endif
	}

	bool FullNLProblem::updateHessianSparsityPattern()
	{
		std::unique_ptr<BCSCHessian> dynamicSparsity;
		const bool force = m_fullSparsityRebuildNeeded;
		if (force) {
			m_hessianSparsity.reset();
			m_hessianSparsityStaticPart.reset();
		}

		bool changed = force;
		bool staticOnly = true;

		for (size_t i = 0; i < forms_.size(); ++i) {
			if (!forms_[i]->enabled()) continue;
			const auto &f = forms_[i];

			if (f->sparsityPatternIsStatic()) {
				// Only rebuild the "static" part when the terms might have been invalidated.
				if (force) {
					// std::cout << "Building static term sparsity pattern" << std::endl;
					auto Hsp = f->hessianSparsityPattern();
					if (!m_hessianSparsityStaticPart) m_hessianSparsityStaticPart = std::move(Hsp);
					else m_hessianSparsityStaticPart->mergeSparsityPattern(Hsp.get());
				}
			}
			else {
				auto Hsp = f->hessianSparsityPattern();
				if (!Hsp) continue; // Skip empty patterns, indicated by a null pointer.
				if (!dynamicSparsity) dynamicSparsity = std::move(Hsp);
				else dynamicSparsity->mergeSparsityPattern(Hsp.get());
				// TODO: use `MeshFEM::SystemAssembler::detectChangedEntries` to
				// avoid unnecessary rebuilds of the dynamic sparsity pattern
				// and calls to `m_sparsityLRU::update`. This requires caching a
				// copy of each dynamic term's pattern, which is generally
				// quite small.
				changed = true;
				staticOnly = false;
			}
		}

		if (changed && (!m_hessianSparsityStaticPart || m_hessianSparsityStaticPart->nnz() == 0)) {
			std::cout << "Warning: Hessian sparsity pattern has no static part; bypassing LRU cache" << std::endl;
			m_hessianSparsity = std::move(dynamicSparsity);
			if (!m_hessianSparsity) std::cout << "Warning: Hessian sparsity pattern is empty" << std::endl;
			return true;
		}

		if (changed) {
			if (force && (m_hessianSparsityStaticPart->numDiagonalBlocks() < m_hessianSparsityStaticPart->m)) {
				// Work around paranoia of the Sparsity LRU object, which checks if all
				// diagonal blocks are present (technically not needed for the specific
				// matrix format used here).
				auto copy = m_hessianSparsityStaticPart->emptyClone();
				copy->setIdentity();
				m_hessianSparsityStaticPart->mergeSparsityPattern(*copy);
			}
			if (staticOnly) m_hessianSparsity = std::move(m_hessianSparsityStaticPart);
			else {
				assert(dynamicSparsity != nullptr);
				if (!m_sparsityLRU){
					m_sparsityLRU = std::make_unique<MeshFEM::SparsityLRU>(*m_hessianSparsityStaticPart);
					m_sparsityLRU->verbose = false;
					m_hessianSparsity = m_hessianSparsityStaticPart->clone();
				}

				changed = m_sparsityLRU->update(*dynamicSparsity);
				changed |= force; // Ensure the static part rebuild takes effect.
				if (changed) {
					if (!m_hessianSparsity) throw std::logic_error("NewtonMultiobjectiveProblem::m_updateSparsityPattern: m_hessianSparsity not initialized"); // This should never happen since `m_sparsityLRU` is only created when the static part is nonempty...
					m_hessianSparsity->Ap = (*m_sparsityLRU)->Ap;
					m_hessianSparsity->Ai = (*m_sparsityLRU)->Ai;
					m_hessianSparsity->nz = (*m_sparsityLRU)->nz;
				}
			}

			m_hessianSparsity->finalize();
		}
		else if (m_sparsityLRU) {
			// Still notify the cache of the sparsity pattern update in case
			// it triggers a re-factorization due to entry expiration.
			if (m_sparsityLRU->increaseAgeOfOldEntries()) {
				if (!m_hessianSparsity) throw std::logic_error("NewtonMultiobjectiveProblem::m_updateSparsityPattern: m_hessianSparsity not initialized"); // This should never happen since `m_sparsityLRU` is only created when the static part is nonempty...
				m_hessianSparsity->Ap = (*m_sparsityLRU)->Ap;
				m_hessianSparsity->Ai = (*m_sparsityLRU)->Ai;
				m_hessianSparsity->nz = (*m_sparsityLRU)->nz;

				m_hessianSparsity->finalize();

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

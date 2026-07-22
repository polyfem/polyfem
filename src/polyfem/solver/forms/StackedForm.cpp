#include "StackedForm.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

namespace polyfem::solver
{
	StackedForm::Block StackedForm::add_block(const int size)
	{
		assert(size > 0);

		const int id = int(blocks_.size());
		Block block(id, size_, size);
		blocks_.push_back(block);
		size_ += size;
		return block;
	}

	void StackedForm::add(const Block &block, std::shared_ptr<Form> form)
	{
		validate_block(block);
		assert(form);
		terms_.push_back({block.id(), -1, std::move(form)});
	}

	void StackedForm::add(const Block &a, const Block &b, std::shared_ptr<Form> form)
	{
		validate_block(a);
		validate_block(b);
		assert(a.id() != b.id());
		assert(form);
		terms_.push_back({a.id(), b.id(), std::move(form)});
	}

	void StackedForm::init(const Eigen::VectorXd &x)
	{
		assert(x.size() == size_);
		for (const Term &term : terms_)
			term.form->init(gather(x, term));
	}

	void StackedForm::finish()
	{
		for (const Term &term : terms_)
			term.form->finish();
	}

	bool StackedForm::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		assert(x0.size() == size_);
		assert(x1.size() == size_);

		for (const Term &term : terms_)
		{
			if (!term.form->enabled())
				continue;
			if (!term.form->is_step_valid(gather(x0, term), gather(x1, term)))
				return false;
		}

		return true;
	}

	bool StackedForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		assert(x0.size() == size_);
		assert(x1.size() == size_);

		for (const Term &term : terms_)
		{
			if (!term.form->enabled())
				continue;
			if (!term.form->is_step_collision_free(gather(x0, term), gather(x1, term)))
				return false;
		}

		return true;
	}

	double StackedForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		assert(x0.size() == size_);
		assert(x1.size() == size_);

		double step = 1;
		for (const Term &term : terms_)
		{
			if (!term.form->enabled())
				continue;
			step = std::min(step, term.form->max_step_size(gather(x0, term), gather(x1, term)));
		}

		return step;
	}

	void StackedForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		assert(x0.size() == size_);
		assert(x1.size() == size_);

		for (const Term &term : terms_)
			term.form->line_search_begin(gather(x0, term), gather(x1, term));
	}

	void StackedForm::line_search_end()
	{
		for (const Term &term : terms_)
			term.form->line_search_end();
	}

	void StackedForm::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		assert(data.x.size() == size_);
		assert(data.grad.size() == size_);

		for (const Term &term : terms_)
		{
			const Eigen::VectorXd x = gather(data.x, term);
			const Eigen::VectorXd grad = gather(data.grad, term);
			const polysolve::nonlinear::PostStepData local_data(data.iter_num, data.solver_info, x, grad);
			term.form->post_step(local_data);
		}
	}

	void StackedForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		assert(new_x.size() == size_);
		for (const Term &term : terms_)
			term.form->solution_changed(gather(new_x, term));
	}

	void StackedForm::set_project_to_psd(const bool val)
	{
		Form::set_project_to_psd(val);
		for (const Term &term : terms_)
			term.form->set_project_to_psd(val);
	}

	void StackedForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		assert(x.size() == size_);
		for (const Term &term : terms_)
			term.form->update_quantities(t, gather(x, term));
	}

	void StackedForm::init_lagging(const Eigen::VectorXd &x)
	{
		assert(x.size() == size_);
		for (const Term &term : terms_)
			term.form->init_lagging(gather(x, term));
	}

	void StackedForm::update_lagging(const Eigen::VectorXd &x, const int iter_num)
	{
		assert(x.size() == size_);
		for (const Term &term : terms_)
			term.form->update_lagging(gather(x, term), iter_num);
	}

	int StackedForm::max_lagging_iterations() const
	{
		int max_lagging_iterations = 1;
		for (const Term &term : terms_)
			max_lagging_iterations = std::max(max_lagging_iterations, term.form->max_lagging_iterations());
		return max_lagging_iterations;
	}

	bool StackedForm::uses_lagging() const
	{
		for (const Term &term : terms_)
		{
			if (term.form->uses_lagging())
				return true;
		}
		return false;
	}

	double StackedForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		assert(x.size() == size_);

		double value = 0;
		for (const Term &term : terms_)
		{
			if (!term.form->enabled())
				continue;
			value += term.form->value(gather(x, term));
		}
		return value;
	}

	void StackedForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(x.size() == size_);

		gradv = Eigen::VectorXd::Zero(size_);
		for (const Term &term : terms_)
		{
			if (!term.form->enabled())
				continue;

			Eigen::VectorXd local_grad;
			term.form->first_derivative(gather(x, term), local_grad);
			assert(local_grad.size() == term_size(term));

			for (int i = 0; i < local_grad.size(); ++i)
				gradv(global_index(term, i)) += local_grad(i);
		}
	}

	void StackedForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		assert(x.size() == size_);

		using StorageIndex = typename StiffnessMatrix::StorageIndex;
		std::vector<Eigen::Triplet<double, StorageIndex>> entries;

		for (const Term &term : terms_)
		{
			if (!term.form->enabled())
				continue;

			StiffnessMatrix local_hessian;
			term.form->second_derivative(gather(x, term), local_hessian);
			const int local_size = term_size(term);
			assert(local_hessian.rows() == local_size);
			assert(local_hessian.cols() == local_size);

			for (int k = 0; k < local_hessian.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(local_hessian, k); it; ++it)
				{
					entries.emplace_back(
						StorageIndex(global_index(term, int(it.row()))),
						StorageIndex(global_index(term, int(it.col()))),
						it.value());
				}
			}
		}

		hessian.resize(size_, size_);
		hessian.setFromTriplets(entries.begin(), entries.end());
	}

	void StackedForm::validate_block(const Block &block) const
	{
		assert(block.id() >= 0);
		assert(block.id() < int(blocks_.size()));
		assert(blocks_[block.id()].offset() == block.offset());
		assert(blocks_[block.id()].size() == block.size());
	}

	int StackedForm::term_size(const Term &term) const
	{
		const Block &a = blocks_[term.a];
		if (!term.has_second_block())
			return a.size();

		const Block &b = blocks_[term.b];
		return a.size() + b.size();
	}

	Eigen::VectorXd StackedForm::gather(const Eigen::VectorXd &x, const Term &term) const
	{
		assert(x.size() == size_);

		const Block &a = blocks_[term.a];
		if (!term.has_second_block())
			return x.segment(a.offset(), a.size());

		const Block &b = blocks_[term.b];
		Eigen::VectorXd local(a.size() + b.size());
		local << x.segment(a.offset(), a.size()), x.segment(b.offset(), b.size());
		return local;
	}

	int StackedForm::global_index(const Term &term, const int local_index) const
	{
		assert(local_index >= 0);
		assert(local_index < term_size(term));

		const Block &a = blocks_[term.a];
		if (!term.has_second_block() || local_index < a.size())
			return a.offset() + local_index;

		const Block &b = blocks_[term.b];
		return b.offset() + local_index - a.size();
	}
} // namespace polyfem::solver

#include "StackedAugmentedLagrangianForm.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

namespace polyfem::solver
{
	StackedAugmentedLagrangianForm::Block StackedAugmentedLagrangianForm::add_block(const int size)
	{
		assert(size > 0);

		const int id = int(blocks_.size());
		Block block(id, size_, size);
		blocks_.push_back(block);
		size_ += size;
		return block;
	}

	void StackedAugmentedLagrangianForm::add(const Block &block, std::shared_ptr<AugmentedLagrangianForm> form)
	{
		validate_block(block);
		assert(form);
		terms_.push_back({block.id(), std::move(form)});
		rebuild_constraint_data();
	}

	void StackedAugmentedLagrangianForm::init(const Eigen::VectorXd &x)
	{
		assert(x.size() == size_);
		sync_child_weights();
		for (const Term &term : terms_)
			term.form->init(gather(x, term));
	}

	void StackedAugmentedLagrangianForm::finish()
	{
		for (const Term &term : terms_)
			term.form->finish();
	}

	void StackedAugmentedLagrangianForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		sync_child_weights();
		for (const Term &term : terms_)
		{
			const Eigen::VectorXd local_x = x.size() == size_ ? gather(x, term) : Eigen::VectorXd();
			term.form->update_quantities(t, local_x);
		}
		rebuild_constraint_data();
	}

	void StackedAugmentedLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
	{
		assert(x.size() == size_);
		k_al_ = k_al;
		for (const Term &term : terms_)
			term.form->update_lagrangian(gather(x, term), k_al);
		rebuild_constraint_data();
	}

	double StackedAugmentedLagrangianForm::compute_error(const Eigen::VectorXd &x) const
	{
		assert(x.size() == size_);

		double error = 0;
		for (const Term &term : terms_)
		{
			if (!term.form->enabled())
				continue;
			error += term.form->compute_error(gather(x, term));
		}
		return error;
	}

	void StackedAugmentedLagrangianForm::project_gradient(Eigen::VectorXd &grad) const
	{
		assert(can_project());
		assert(grad.size() == size_);
		grad = A_proj_.transpose() * grad;
	}

	void StackedAugmentedLagrangianForm::project_hessian(StiffnessMatrix &hessian) const
	{
		assert(can_project());
		assert(hessian.rows() == size_ && hessian.cols() == size_);
		hessian = A_proj_.transpose() * hessian * A_proj_;
		hessian.prune([](const Eigen::Index &, const Eigen::Index &, const double &value) {
			return std::abs(value) > 1e-10;
		});
	}

	void StackedAugmentedLagrangianForm::project_diag(Eigen::VectorXd &diag) const
	{
		assert(can_project());
		assert(diag.size() == size_);
		const StiffnessMatrix reduced = A_proj_.transpose() * diag.asDiagonal() * A_proj_;
		diag = reduced.diagonal();
	}

	double StackedAugmentedLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		assert(x.size() == size_);
		sync_child_weights();

		double value = 0;
		for (const Term &term : terms_)
		{
			if (!term.form->enabled())
				continue;
			value += term.form->value(gather(x, term));
		}
		return value;
	}

	void StackedAugmentedLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(x.size() == size_);
		sync_child_weights();

		gradv = Eigen::VectorXd::Zero(size_);
		for (const Term &term : terms_)
		{
			if (!term.form->enabled())
				continue;

			Eigen::VectorXd local_grad;
			term.form->first_derivative(gather(x, term), local_grad);
			assert(local_grad.size() == blocks_[term.block].size());

			for (int i = 0; i < local_grad.size(); ++i)
				gradv(global_index(term, i)) += local_grad(i);
		}
	}

	void StackedAugmentedLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		assert(x.size() == size_);
		sync_child_weights();

		using StorageIndex = typename StiffnessMatrix::StorageIndex;
		std::vector<Eigen::Triplet<double, StorageIndex>> entries;

		for (const Term &term : terms_)
		{
			if (!term.form->enabled())
				continue;

			StiffnessMatrix local_hessian;
			term.form->second_derivative(gather(x, term), local_hessian);
			const int local_size = blocks_[term.block].size();
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

	void StackedAugmentedLagrangianForm::validate_block(const Block &block) const
	{
		assert(block.id() >= 0);
		assert(block.id() < int(blocks_.size()));
		assert(blocks_[block.id()].offset() == block.offset());
		assert(blocks_[block.id()].size() == block.size());
	}

	Eigen::VectorXd StackedAugmentedLagrangianForm::gather(const Eigen::VectorXd &x, const Term &term) const
	{
		assert(x.size() == size_);
		const Block &block = blocks_[term.block];
		return x.segment(block.offset(), block.size());
	}

	int StackedAugmentedLagrangianForm::global_index(const Term &term, const int local_index) const
	{
		assert(local_index >= 0);
		assert(local_index < blocks_[term.block].size());
		return blocks_[term.block].offset() + local_index;
	}

	void StackedAugmentedLagrangianForm::rebuild_constraint_data()
	{
		int constraint_rows = 0;
		for (const Term &term : terms_)
		{
			assert(term.form);
			assert(term.form->constraint_matrix().cols() == blocks_[term.block].size());
			assert(term.form->constraint_value().rows() == term.form->constraint_matrix().rows());
			constraint_rows += term.form->constraint_matrix().rows();
		}

		std::vector<Eigen::Triplet<double>> constraint_entries;
		int constraint_offset = 0;
		b_.resize(constraint_rows, 1);
		for (const Term &term : terms_)
		{
			const Block &block = blocks_[term.block];
			const StiffnessMatrix &local_A = term.form->constraint_matrix();

			for (int k = 0; k < local_A.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(local_A, k); it; ++it)
				{
					constraint_entries.emplace_back(
						constraint_offset + int(it.row()),
						block.offset() + int(it.col()),
						it.value());
				}
			}

			b_.middleRows(constraint_offset, local_A.rows()) = term.form->constraint_value();
			constraint_offset += local_A.rows();
		}

		A_.resize(constraint_rows, size_);
		A_.setFromTriplets(constraint_entries.begin(), constraint_entries.end());
		A_.makeCompressed();

		std::vector<int> term_for_block(blocks_.size(), -1);
		projection_available_ = !terms_.empty();
		for (int i = 0; i < int(terms_.size()); ++i)
		{
			const Term &term = terms_[i];
			if (term_for_block[term.block] >= 0 || !term.form->has_projection())
			{
				projection_available_ = false;
				break;
			}
			term_for_block[term.block] = i;
		}

		if (!projection_available_)
		{
			A_proj_.resize(0, 0);
			b_proj_.resize(0, 0);
			return;
		}

		int reduced_size = 0;
		for (int i = 0; i < int(blocks_.size()); ++i)
		{
			if (term_for_block[i] >= 0)
				reduced_size += terms_[term_for_block[i]].form->constraint_projection_matrix().cols();
			else
				reduced_size += blocks_[i].size();
		}

		std::vector<Eigen::Triplet<double>> projection_entries;
		b_proj_ = Eigen::VectorXd::Zero(size_);
		int reduced_offset = 0;
		for (int i = 0; i < int(blocks_.size()); ++i)
		{
			const Block &block = blocks_[i];
			const int term_id = term_for_block[i];

			if (term_id < 0)
			{
				for (int j = 0; j < block.size(); ++j)
					projection_entries.emplace_back(block.offset() + j, reduced_offset + j, 1.0);
				reduced_offset += block.size();
				continue;
			}

			const auto &form = terms_[term_id].form;
			const StiffnessMatrix &local_projection = form->constraint_projection_matrix();
			const Eigen::MatrixXd &local_projection_value = form->constraint_projection_vector();
			assert(local_projection.rows() == block.size());
			assert(local_projection_value.rows() == block.size());

			for (int k = 0; k < local_projection.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(local_projection, k); it; ++it)
				{
					projection_entries.emplace_back(
						block.offset() + int(it.row()),
						reduced_offset + int(it.col()),
						it.value());
				}
			}

			b_proj_.middleRows(block.offset(), block.size()) = local_projection_value;
			reduced_offset += local_projection.cols();
		}

		A_proj_.resize(size_, reduced_size);
		A_proj_.setFromTriplets(projection_entries.begin(), projection_entries.end());
		A_proj_.makeCompressed();
	}

	void StackedAugmentedLagrangianForm::sync_child_weights() const
	{
		for (const Term &term : terms_)
			term.form->set_initial_weight(k_al_);
	}
} // namespace polyfem::solver

#pragma once

#include "AugmentedLagrangianForm.hpp"

#include <memory>
#include <vector>

namespace polyfem::solver
{
	class StackedAugmentedLagrangianForm : public AugmentedLagrangianForm
	{
	public:
		class Block
		{
			friend class StackedAugmentedLagrangianForm;

		public:
			int id() const { return id_; }
			int offset() const { return offset_; }
			int size() const { return size_; }

		private:
			Block(const int id, const int offset, const int size)
				: id_(id), offset_(offset), size_(size) {}

			int id_ = -1;
			int offset_ = 0;
			int size_ = 0;
		};

		StackedAugmentedLagrangianForm() { k_al_ = 0; }

		std::string name() const override { return "stacked-alagrangian"; }

		Block add_block(const int size);
		void add(const Block &block, std::shared_ptr<AugmentedLagrangianForm> form);

		int size() const { return size_; }

		void init(const Eigen::VectorXd &x) override;
		void finish() override;
		void update_quantities(const double t, const Eigen::VectorXd &x) override;

		void update_lagrangian(const Eigen::VectorXd &x, const double k_al) override;
		double compute_error(const Eigen::VectorXd &x) const override;

		bool can_project() const override { return has_projection(); }
		void project_gradient(Eigen::VectorXd &grad) const override;
		void project_hessian(StiffnessMatrix &hessian) const override;
		void project_diag(Eigen::VectorXd &diag) const override;

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override;
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:
		struct Term
		{
			int block = -1;
			std::shared_ptr<AugmentedLagrangianForm> form;
		};

		void validate_block(const Block &block) const;
		Eigen::VectorXd gather(const Eigen::VectorXd &x, const Term &term) const;
		int global_index(const Term &term, const int local_index) const;
		void rebuild_constraint_data();
		void sync_child_weights() const;

		std::vector<Block> blocks_;
		std::vector<Term> terms_;
		int size_ = 0;
		bool projection_available_ = false;
	};
} // namespace polyfem::solver

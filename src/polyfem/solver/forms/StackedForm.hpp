#pragma once

#include "Form.hpp"

#include <memory>
#include <vector>

namespace polyfem::solver
{
	class StackedForm : public Form
	{
	public:
		class Block
		{
			friend class StackedForm;

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

		std::string name() const override { return "stacked"; }

		Block add_block(const int size);
		void add(const Block &block, std::shared_ptr<Form> form);
		void add(const Block &a, const Block &b, std::shared_ptr<Form> form);

		int size() const { return size_; }

		void init(const Eigen::VectorXd &x) override;
		void finish() override;
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;
		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		void line_search_end() override;
		void post_step(const polysolve::nonlinear::PostStepData &data) override;
		void solution_changed(const Eigen::VectorXd &new_x) override;
		void update_quantities(const double t, const Eigen::VectorXd &x) override;
		void init_lagging(const Eigen::VectorXd &x) override;
		void update_lagging(const Eigen::VectorXd &x, const int iter_num) override;
		int max_lagging_iterations() const override;
		bool uses_lagging() const override;

	protected:
		double value_unweighted(const Eigen::VectorXd &x) const override;
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:
		struct Term
		{
			int a = -1;
			int b = -1;
			std::shared_ptr<Form> form;

			bool has_second_block() const { return b >= 0; }
		};

		void validate_block(const Block &block) const;
		int term_size(const Term &term) const;
		Eigen::VectorXd gather(const Eigen::VectorXd &x, const Term &term) const;
		int global_index(const Term &term, const int local_index) const;

		std::vector<Block> blocks_;
		std::vector<Term> terms_;
		int size_ = 0;
	};
} // namespace polyfem::solver

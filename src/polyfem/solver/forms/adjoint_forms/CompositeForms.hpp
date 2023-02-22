#pragma once

#include "CompositeForm.hpp"

namespace polyfem::solver
{
	class HomoCompositeForm : public CompositeForm
	{
	public:
		using CompositeForm::CompositeForm;
		~HomoCompositeForm() {}
	
	private:
		double compose(const Eigen::VectorXd &inputs) const override;
		Eigen::VectorXd compose_grad(const Eigen::VectorXd &inputs) const override;
	};

	class NegativeCompositeForm : public CompositeForm
	{
	public:
		NegativeCompositeForm(const std::shared_ptr<AdjointForm> &form) : CompositeForm({form}) {}
		~NegativeCompositeForm() {}
	
	private:
		double compose(const Eigen::VectorXd &inputs) const override { assert(inputs.size() == 1); return -inputs(0); }
		Eigen::VectorXd compose_grad(const Eigen::VectorXd &inputs) const override { assert(inputs.size() == 1); return Eigen::VectorXd::Constant(1, 1, -1.); }
	};

	class PlusConstCompositeForm : public CompositeForm
	{
	public:
		PlusConstCompositeForm(const std::shared_ptr<AdjointForm> &form, const double alpha) : CompositeForm({form}), alpha_(alpha) {}
		~PlusConstCompositeForm() {}
	
	private:
		const double alpha_;

		double compose(const Eigen::VectorXd &inputs) const override { assert(inputs.size() == 1); return alpha_ + inputs(0); }
		Eigen::VectorXd compose_grad(const Eigen::VectorXd &inputs) const override { assert(inputs.size() == 1); return Eigen::VectorXd::Constant(1, 1, 1.); }
	};

	class InequalityConstraintForm : public CompositeForm
	{
	public:
		InequalityConstraintForm(const std::vector<std::shared_ptr<AdjointForm>> &forms, const Eigen::Vector2d &bounds);
		~InequalityConstraintForm() {}
	
	private:
		double compose(const Eigen::VectorXd &inputs) const override;
		Eigen::VectorXd compose_grad(const Eigen::VectorXd &inputs) const override;

		const Eigen::Vector2d bounds_;
	};
}
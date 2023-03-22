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
		InequalityConstraintForm(const std::vector<std::shared_ptr<AdjointForm>> &forms, const Eigen::Vector2d &bounds, const double power = 2);
		~InequalityConstraintForm() {}
	
	private:
		double compose(const Eigen::VectorXd &inputs) const override;
		Eigen::VectorXd compose_grad(const Eigen::VectorXd &inputs) const override;

		const double power_;
		const Eigen::Vector2d bounds_;
	};

	class PowerForm : public CompositeForm
	{
	public:
		PowerForm(const std::shared_ptr<AdjointForm> &form, const double power): CompositeForm({form}), power_(power) { }
		~PowerForm() {}
	
	private:
		double compose(const Eigen::VectorXd &inputs) const override { return pow(inputs(0), power_); }
		Eigen::VectorXd compose_grad(const Eigen::VectorXd &inputs) const override { Eigen::VectorXd x(1); x(0) = power_ * pow(inputs(0), power_ - 1); return x; }

		const double power_;
	};

	class DivideForm : public CompositeForm
	{
	public:
		DivideForm(const std::vector<std::shared_ptr<AdjointForm>> &forms): CompositeForm(forms) { assert(forms.size() == 2); }
		~DivideForm() {}
	
	private:
		double compose(const Eigen::VectorXd &inputs) const override { return inputs(0) / inputs(1); }
		Eigen::VectorXd compose_grad(const Eigen::VectorXd &inputs) const override { Eigen::VectorXd x(2); x << 1. / inputs(1), -inputs(0) / pow(inputs(1), 2); return x; }
	};
}
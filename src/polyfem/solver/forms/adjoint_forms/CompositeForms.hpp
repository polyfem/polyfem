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
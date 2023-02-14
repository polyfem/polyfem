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
}
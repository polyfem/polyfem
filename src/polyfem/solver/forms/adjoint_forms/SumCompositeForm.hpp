#include "CompositeForm.hpp"

namespace polyfem::solver
{
	class SumCompositeForm : public CompositeForm
	{
		using CompositeForm::CompositeForm;

		inline double compose(const Eigen::VectorXd &inputs) const override
		{
			return inputs.sum();
		}

		inline Eigen::VectorXd compose_grad(const Eigen::VectorXd &inputs) const override
		{
			return Eigen::VectorXd::Ones(inputs.size());
		}
	};
} // namespace polyfem::solver
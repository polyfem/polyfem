#include "CompositeForm.hpp"

namespace polyfem::solver
{
	class SumCompositeForm : public CompositeForm
	{
		using CompositeForm::CompositeForm;
	public:
		SumCompositeForm(const std::vector<std::shared_ptr<VariableToSimulation>> &variable_to_simulations, const CompositeParametrization &parametrizations, const std::vector<std::shared_ptr<ParametrizationForm>> &forms) : CompositeForm(variable_to_simulations, parametrizations, forms) {}
		~SumCompositeForm() {}
	
	private:
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
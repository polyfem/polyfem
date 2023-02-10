#include "CompositeForm.hpp"

namespace polyfem::solver
{
	class SumCompositeForm : public CompositeForm
	{
		inline double compose(const double first_form_output, const double second_form_output) override
		{
			return first_form_output + second_form_output;
		}

		inline Eigen::VectorXd compose_derivative(const Eigen::VectorXd &first_form_derivative, const Eigen::VectorXd &second_form_derivative) override
		{
			return first_form_derivative + second_form_derivative;
		}
	}
} // namespace polyfem::solver
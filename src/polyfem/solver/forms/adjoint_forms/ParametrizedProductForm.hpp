#pragma once

#include <polyfem/solver/forms/adjoint_forms/ParametrizationForm.hpp>
#include "VariableToSimulation.hpp"

namespace polyfem::solver
{
	// computes the product of a parametrized vector
	class ParametrizedProductForm : public ParametrizationForm
	{
	public:
		ParametrizedProductForm(const CompositeParametrization &parametrizations) : ParametrizationForm(parametrizations)
		{
		}

		inline double value_unweighted_with_param(const Eigen::VectorXd &x) const override
		{
			return x.prod();
		}

		inline void first_derivative_unweighted_with_param(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
		{
			gradv.setOnes(x.size());
			for (int i = 0; i < x.size(); i++)
			{
				for (int j = 0; j < x.size(); j++)
				{
					if (j != i)
						gradv(i) *= x(j);
				}
			}
		}
	};
} // namespace polyfem::solver
#pragma once

#include <polyfem/State.hpp>
#include <polyfem/solver/forms/Form.hpp>

namespace polyfem::solver
{
	class AdjointForm : public Form
	{
	public:
		AdjointForm(const State &state) : state_(state){};

		inline double value(const Eigen::VectorXd &x) const override
		{
			return 0;
		}

		inline void first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
		{
		}

		inline void second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override
		{
			logger().error("Called invalid second derivative function.");
		}

	private:
		const State &state_;
	};
} // namespace polyfem::solver

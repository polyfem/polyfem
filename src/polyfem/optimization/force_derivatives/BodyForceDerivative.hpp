#pragma once

#include <Eigen/Core>
#include <polyfem/solver/forms/BodyForm.hpp>

namespace polyfem::solver
{
	class BodyForceDerivative
	{
	public:
		static void force_shape_derivative(
			BodyForm &form,
			const int n_verts,
			const double t,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &term);
	};
} // namespace polyfem::solver

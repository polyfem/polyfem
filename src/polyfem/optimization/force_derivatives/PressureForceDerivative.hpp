#pragma once

#include <Eigen/Core>
#include <polyfem/solver/forms/PressureForm.hpp>

namespace polyfem::solver
{
	class PressureForceDerivative
	{
	public:
		static void force_shape_derivative(
			PressureForm &form,
			const int n_verts,
			const double t,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &term);

		static double force_pressure_derivative(
			PressureForm &form,
			const int n_verts,
			const double t,
			const int pressure_boundary_id,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &adjoint);
	};
} // namespace polyfem::solver

#pragma once

#include <Eigen/Core>
#include <polyfem/solver/forms/ElasticForm.hpp>

namespace polyfem::solver
{
	class ElasticForceDerivative
	{
	public:
		static void force_material_derivative(
			ElasticForm &form,
			const double t,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &x_prev,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &term);

		static void force_shape_derivative(
			ElasticForm &form,
			const double t,
			const int n_verts,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &x_prev,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &term);
	};
} // namespace polyfem::solver

#pragma once

#include <Eigen/Core>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <ipc/collisions/tangential/tangential_collisions.hpp>

namespace polyfem::solver
{
	class FrictionForceDerivative
	{
	public:
		static void force_shape_derivative(
			FrictionForm &form,
			const Eigen::MatrixXd &prev_solution,
			const Eigen::MatrixXd &solution,
			const Eigen::MatrixXd &adjoint,
			const ipc::TangentialCollisions &friction_constraints_set,
			Eigen::VectorXd &term);
	};
} // namespace polyfem::solver

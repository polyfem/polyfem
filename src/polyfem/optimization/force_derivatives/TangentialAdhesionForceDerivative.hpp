#pragma once

#include <Eigen/Core>
#include <polyfem/solver/forms/TangentialAdhesionForm.hpp>
#include <ipc/collisions/tangential/tangential_collisions.hpp>

namespace polyfem::solver
{
	class TangentialAdhesionForceDerivative
	{
	public:
		static void force_shape_derivative(
			TangentialAdhesionForm &form,
			const Eigen::MatrixXd &prev_solution,
			const Eigen::MatrixXd &solution,
			const Eigen::MatrixXd &adjoint,
			const ipc::TangentialCollisions &tangential_constraints_set,
			Eigen::VectorXd &term);
	};
} // namespace polyfem::solver

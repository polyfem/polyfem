#pragma once

#include <Eigen/Core>
#include <polyfem/solver/forms/NormalAdhesionForm.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>

namespace polyfem::solver
{
	class NormalAdhesionForceDerivative
	{
	public:
		static void force_shape_derivative(
			NormalAdhesionForm &form,
			const ipc::NormalCollisions &collision_set,
			const Eigen::MatrixXd &solution,
			const Eigen::VectorXd &adjoint_sol,
			Eigen::VectorXd &term);
	};
} // namespace polyfem::solver

#pragma once

#include <Eigen/Core>
#include <polyfem/solver/forms/SmoothContactForm.hpp>
#include <ipc/gcp/gcp_collisions.hpp>

namespace polyfem::solver
{
	class SmoothContactForceDerivative
	{
	public:
		static void force_shape_derivative(
			const SmoothContactForm &form,
			const ipc::GCPCollisions &collision_set,
			const Eigen::MatrixXd &solution,
			const Eigen::VectorXd &adjoint_sol,
			Eigen::VectorXd &term);
	};
} // namespace polyfem::solver

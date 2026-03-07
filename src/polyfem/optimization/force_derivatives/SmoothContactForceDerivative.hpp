#pragma once

#include <Eigen/Core>
#include <polyfem/solver/forms/SmoothContactForm.hpp>
#include <ipc/smooth_contact/smooth_collisions.hpp>

namespace polyfem::solver
{
	class SmoothContactForceDerivative
	{
	public:
		static void force_shape_derivative(
			const SmoothContactForm &form,
			const ipc::SmoothCollisions &collision_set,
			const Eigen::MatrixXd &solution,
			const Eigen::VectorXd &adjoint_sol,
			Eigen::VectorXd &term);
	};
} // namespace polyfem::solver

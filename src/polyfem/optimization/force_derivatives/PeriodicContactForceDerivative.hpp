#pragma once

#include <Eigen/Core>
#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/State.hpp>
#include <polyfem/optimization/parametrization/PeriodicMeshToMesh.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>

namespace polyfem::solver
{
	class PeriodicContactForceDerivative
	{
	public:
		static void force_shape_derivative(
			const PeriodicContactForm &form,
			const State &state,
			const PeriodicMeshToMesh &periodic_mesh_map,
			const Eigen::VectorXd &periodic_mesh_representation,
			const ipc::NormalCollisions &contact_set,
			const Eigen::VectorXd &solution,
			const Eigen::VectorXd &adjoint_sol,
			Eigen::VectorXd &term);
	};
} // namespace polyfem::solver

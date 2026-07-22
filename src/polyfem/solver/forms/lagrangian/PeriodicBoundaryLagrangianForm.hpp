#pragma once

#include "MatrixLagrangianForm.hpp"

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/mesh/Mesh.hpp>

namespace polyfem::solver
{
	/// Linear equality constraints coupling corresponding DoFs on two tagged boundaries.
	class PeriodicBoundaryLagrangianForm : public MatrixLagrangianForm
	{
	public:
		PeriodicBoundaryLagrangianForm(
			int ndof,
			int value_dim,
			const mesh::Mesh &mesh,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<mesh::LocalBoundary> &local_boundary,
			const std::array<int, 2> &boundary_ids,
			double relative_tolerance);

		std::string name() const override { return "periodic-boundary-lagrangian"; }

	private:
		struct ConstraintData
		{
			StiffnessMatrix A;
			Eigen::MatrixXd b;
		};

		explicit PeriodicBoundaryLagrangianForm(ConstraintData data);

		static ConstraintData build_constraints(
			int ndof,
			int value_dim,
			const mesh::Mesh &mesh,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<mesh::LocalBoundary> &local_boundary,
			const std::array<int, 2> &boundary_ids,
			double relative_tolerance);
	};
} // namespace polyfem::solver

#pragma once

#include <Eigen/Core>

#include <vector>

#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>

namespace polyfem
{

	/// @brief Select interior nodes (vertex id).
	/// @param varform Forward sim varform.
	/// @param volume_selection Body ID to select. Empty implies all.
	Eigen::VectorXi select_interior_nodes(
		const varform::DifferentiableVarForm &varform,
		const std::vector<int> &volume_selection);

	/// @brief Select boundary nodes (vertex id).
	/// @param varform Forward sim varform.
	/// @param surface_selection Boundary ID to select. Empty implies all.
	Eigen::VectorXi select_boundary_nodes(
		const varform::DifferentiableVarForm &varform,
		const std::vector<int> &surface_selection);

	/// @brief Select all boundary nodes (vertex id) except surface.
	/// @param varform Forward sim varform.
	/// @param exclude_surface_selections Boundary ID to exclude. Empty implies none.
	Eigen::VectorXi select_boundary_nodes_excluding_surfaces(
		const varform::DifferentiableVarForm &varform,
		const std::vector<int> &exclude_surface_selections);

} // namespace polyfem

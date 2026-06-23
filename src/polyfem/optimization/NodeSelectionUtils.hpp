#pragma once

#include <Eigen/Core>

#include <vector>

#include <polyfem/legacy/State.hpp>

namespace polyfem
{

	/// @brief Select interior nodes (vertex id).
	/// @param state Forward sim state.
	/// @param volume_selection Body ID to select. Empty implies all.
	Eigen::VectorXi select_interior_nodes(
		const legacy::State &state,
		const std::vector<int> &volume_selection);

	/// @brief Select boundary nodes (vertex id).
	/// @param state Forward sim state.
	/// @param surface_selection Boundary ID to select. Empty implies all.
	Eigen::VectorXi select_boundary_nodes(
		const legacy::State &state,
		const std::vector<int> &surface_selection);

	/// @brief Select all boundary nodes (vertex id) except surface.
	/// @param state Forward sim state.
	/// @param exclude_surface_selections Boundary ID to exclude. Empty implies none.
	Eigen::VectorXi select_boundary_nodes_excluding_surfaces(
		const legacy::State &state,
		const std::vector<int> &exclude_surface_selections);

} // namespace polyfem

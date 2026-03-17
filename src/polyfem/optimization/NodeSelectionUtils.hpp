#pragma once

#include <Eigen/Core>

#include <vector>

#include <polyfem/State.hpp>

namespace polyfem
{

	// Helpers to select geometry nodes (vertex id).

	Eigen::VectorXi select_interior_nodes(
		const State &state,
		const std::vector<int> &volume_selection);

	Eigen::VectorXi select_boundary_nodes(
		const State &state,
		const std::vector<int> &surface_selection);

	Eigen::VectorXi select_boundary_nodes_excluding_surfaces(
		const State &state,
		const std::vector<int> &exclude_surface_selections);

} // namespace polyfem

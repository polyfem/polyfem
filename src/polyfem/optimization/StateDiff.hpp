#pragma once

#include <polyfem/State.hpp>
#include <polyfem/optimization/DiffCache.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>
#include <vector>

namespace polyfem
{
	// Solves the adjoint PDE for derivatives and caches
	void solve_adjoint_cached(const State &state, DiffCache &diff_cache, const Eigen::MatrixXd &rhs);

	// Returns cached adjoint solve
	Eigen::MatrixXd get_adjoint_mat(const State &state, const DiffCache &diff_cache, int type);

	// Get geometric node indices for surface/volume
	void compute_surface_node_ids(const State &state, const int surface_selection, std::vector<int> &node_ids);
	void compute_total_surface_node_ids(const State &state, std::vector<int> &node_ids);
	void compute_volume_node_ids(const State &state, const int volume_selection, std::vector<int> &node_ids);
} // namespace polyfem

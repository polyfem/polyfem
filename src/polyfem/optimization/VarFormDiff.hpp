#pragma once

#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>
#include <polyfem/optimization/DiffCache.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>
#include <vector>

namespace polyfem
{
	// Solves the adjoint PDE for derivatives and caches
	void solve_adjoint_cached(const varform::DifferentiableVarForm &varform, DiffCache &diff_cache, const Eigen::MatrixXd &rhs);

	// Returns cached adjoint solve
	Eigen::MatrixXd get_adjoint_mat(const varform::DifferentiableVarForm &varform, const DiffCache &diff_cache, int type);

	// Get geometric node indices for surface/volume
	void compute_surface_node_ids(const varform::DifferentiableVarForm &varform, const int surface_selection, std::vector<int> &node_ids);
	void compute_total_surface_node_ids(const varform::DifferentiableVarForm &varform, std::vector<int> &node_ids);
	void compute_volume_node_ids(const varform::DifferentiableVarForm &varform, const int volume_selection, std::vector<int> &node_ids);
} // namespace polyfem

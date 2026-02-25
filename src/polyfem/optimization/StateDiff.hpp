#pragma once

#include <polyfem/State.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>
#include <vector>

namespace polyfem
{
	void cache_transient_adjoint_quantities(State &state, const int current_step, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad);

	// Aux functions for setting up adjoint equations
	void compute_force_jacobian(State &state, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &disp_grad, StiffnessMatrix &hessian);
	void compute_force_jacobian_prev(const State &state, const int force_step, const int sol_step, StiffnessMatrix &hessian_prev);

	// Solves the adjoint PDE for derivatives and caches
	void solve_adjoint_cached(State &state, const Eigen::MatrixXd &rhs);
	Eigen::MatrixXd solve_adjoint(const State &state, const Eigen::MatrixXd &rhs);

	// Returns cached adjoint solve
	Eigen::MatrixXd get_adjoint_mat(const State &state, int type);

	Eigen::MatrixXd solve_static_adjoint(const State &state, const Eigen::MatrixXd &adjoint_rhs);
	Eigen::MatrixXd solve_transient_adjoint(const State &state, const Eigen::MatrixXd &adjoint_rhs);

	// Get geometric node indices for surface/volume
	void compute_surface_node_ids(const State &state, const int surface_selection, std::vector<int> &node_ids);
	void compute_total_surface_node_ids(const State &state, std::vector<int> &node_ids);
	void compute_volume_node_ids(const State &state, const int volume_selection, std::vector<int> &node_ids);
} // namespace polyfem

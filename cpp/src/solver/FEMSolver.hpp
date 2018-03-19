#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Common.hpp"
#include "LinearSolver.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
////////////////////////////////////////////////////////////////////////////////

namespace poly_fem {

///
/// @brief         { Solve a linear system Ax = b with Dirichlet boundary
///                conditions. For each Dirichlet node i, we want to ensure that
///                x[i] = b[i]. The implementation follows
///                http://www.math.colostate.edu/~bangerth/videos.676.21.65.html
///
///                For memory efficiency, this function updates in place the
///                matrix of the linear system. We also return the modified rhs
///                to the user }
///
/// @param[in]     solver           { Linear solver class to use for solving the
///                                 system }
/// @param[in,out] A                { Matrix of the linear system without
///                                 boundary conditions. Output: modified
///                                 matrix. }
/// @param[in,out] b                { Right-hand side of the linear system.
///                                 Output: modified rhs for the equivalent
///                                 system. }
/// @param[in]     dirichlet_nodes  { List of ids of Dirichlet nodes }
/// @param[in,out] x                { Unknown vector }
///
void dirichlet_solve(LinearSolver &solver, Eigen::SparseMatrix<double> &A,
	Eigen::VectorXd &b, const std::vector<int> &dirichlet_nodes, Eigen::VectorXd &x, const bool analyze_pattern = true);

} // namespace poly_fem

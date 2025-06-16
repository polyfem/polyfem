#pragma once

#include <polyfem/assembler/MassMatrixAssembler.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/BarrierContactForm.hpp>

#include <polysolve/nonlinear/Solver.hpp>

namespace polyfem::mesh
{
	Eigen::MatrixXd unconstrained_L2_projection(
		const Eigen::SparseMatrix<double> &M,
		const Eigen::SparseMatrix<double> &A,
		const Eigen::Ref<const Eigen::MatrixXd> &y);

	void reduced_L2_projection(
		const Eigen::MatrixXd &M,
		const Eigen::MatrixXd &A,
		const Eigen::Ref<const Eigen::MatrixXd> &y,
		const std::vector<int> &boundary_nodes,
		Eigen::Ref<Eigen::MatrixXd> x);

	Eigen::VectorXd constrained_L2_projection(
		// Nonlinear solver
		std::shared_ptr<polysolve::nonlinear::Solver> nl_solver,
		// L2 projection form
		const Eigen::SparseMatrix<double> &M,
		const Eigen::SparseMatrix<double> &A,
		const Eigen::VectorXd &y,
		// Inversion-free form
		const Eigen::MatrixXd &rest_positions,
		const Eigen::MatrixXi &elements,
		const int dim,
		// Contact form
		const ipc::CollisionMesh &collision_mesh,
		const double dhat,
		const double barrier_stiffness,
		const bool use_area_weighting,
		const bool use_improved_max_operator,
		const bool use_physical_barrier,
		const ipc::BroadPhaseMethod broad_phase_method,
		const double ccd_tolerance,
		const int ccd_max_iterations,
		// Augmented lagrangian form
		const std::vector<int> &boundary_nodes,
		const size_t obstacle_ndof,
		const Eigen::VectorXd &target_x,
		// Initial guess
		const Eigen::VectorXd &x0);

} // namespace polyfem::mesh
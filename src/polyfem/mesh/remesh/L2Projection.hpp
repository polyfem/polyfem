#pragma once

#include <polyfem/assembler/MassMatrixAssembler.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>

namespace polyfem::mesh
{
	Eigen::MatrixXd unconstrained_L2_projection(
		const Eigen::SparseMatrix<double> &M,
		const Eigen::SparseMatrix<double> &A,
		const Eigen::MatrixXd &y);

	Eigen::VectorXd constrained_L2_projection(
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
		const bool use_convergent_formulation,
		const ipc::BroadPhaseMethod broad_phase_method,
		const double ccd_tolerance,
		const int ccd_max_iterations,
		// Augmented lagrangian form
		const std::vector<int> &boundary_nodes,
		const Obstacle &obstacle,
		const Eigen::VectorXd &target_x,
		// Initial guess
		const Eigen::VectorXd &x0);

	class StaticBoundaryNLProblem : public polyfem::solver::NLProblem
	{
	public:
		StaticBoundaryNLProblem(
			const int full_size,
			const std::vector<int> &boundary_nodes,
			const Eigen::VectorXd &boundary_values,
			const std::vector<std::shared_ptr<polyfem::solver::Form>> &forms)
			: polyfem::solver::NLProblem(full_size, boundary_nodes, forms),
			  boundary_values_(boundary_values)
		{
		}

	protected:
		Eigen::MatrixXd boundary_values() const override { return boundary_values_; }

	private:
		const Eigen::MatrixXd boundary_values_;
	};

} // namespace polyfem::mesh
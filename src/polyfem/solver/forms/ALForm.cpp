#include "ALForm.hpp"

namespace polyfem::solver
{
	ALForm::ALForm(const State &state, const assembler::RhsAssembler &rhs_assembler, const double t)
		: state_(state), rhs_assembler_(rhs_assembler), enabled_(true)
	{
		const int ndof = state.n_bases * state.mesh->dimension();

		std::vector<bool> is_boundary_dof(ndof, true);
		for (const auto bn : state.boundary_nodes)
			is_boundary_dof[bn] = false;

		masked_lumped_mass_ = state.mass.size() == 0 ? polyfem::utils::sparse_identity(ndof, ndof) : polyfem::utils::lump_matrix(state.mass);
		assert(ndof == masked_lumped_mass_.rows() && ndof == masked_lumped_mass_.cols());

		// Give the collision obstacles a entry in the lumped mass matrix
		if (state.obstacle.n_vertices() != 0)
		{
			const int n_fe_dof = ndof - state.obstacle.ndof();
			const double avg_mass = masked_lumped_mass_.diagonal().head(n_fe_dof).mean();
			for (int i = n_fe_dof; i < ndof; ++i)
			{
				masked_lumped_mass_.coeffRef(i, i) = avg_mass;
			}
		}

		// Remove non-boundary ndof from mass matrix
		masked_lumped_mass_.prune([&](const int &row, const int &col, const double &value) -> bool {
			assert(row == col); // matrix should be diagonal
			return !is_boundary_dof[row];
		});

		update_target(t);
	}

	double ALForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd dist = x - target_x_;
		const double AL_penalty = 0.5 * dist.transpose() * masked_lumped_mass_ * dist;

		// TODO: Implement Lagrangian potential if needed (i.e., penalty exceeds maximum)
		// ₙ    __
		// ∑ -⎷ mₖ λₖᵀ (xₖ - x̂ₖ) = -λᵀ M (x - x̂)
		// ᵏ

		logger().trace("AL_penalty={}", sqrt(AL_penalty));

		return AL_penalty;
	}

	void ALForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = masked_lumped_mass_ * (x - target_x_);
	}

	void ALForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		hessian = masked_lumped_mass_;
	}

	void ALForm::update_quantities(const double t, const Eigen::VectorXd &)
	{
		if (state_.problem->is_time_dependent())
			update_target(t);
	}

	void ALForm::update_target(const double t)
	{
		target_x_.setZero(masked_lumped_mass_.rows(), 1);
		rhs_assembler_.set_bc(state_.local_boundary, state_.boundary_nodes, state_.n_boundary_samples(), state_.local_neumann_boundary, target_x_, t);
	}
} // namespace polyfem::solver

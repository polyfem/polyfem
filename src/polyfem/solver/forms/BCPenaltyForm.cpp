#include "BCPenaltyForm.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	BCPenaltyForm::BCPenaltyForm(const int ndof,
								 const std::vector<int> &boundary_nodes,
								 const std::vector<mesh::LocalBoundary> &local_boundary,
								 const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
								 const int n_boundary_samples,
								 const StiffnessMatrix &mass,
								 const assembler::RhsAssembler &rhs_assembler,
								 const size_t obstacle_ndof,
								 const bool is_time_dependent,
								 const double t)
		: boundary_nodes_(boundary_nodes),
		  local_boundary_(&local_boundary),
		  local_neumann_boundary_(&local_neumann_boundary),
		  n_boundary_samples_(n_boundary_samples),
		  rhs_assembler_(&rhs_assembler),
		  is_time_dependent_(is_time_dependent)
	{
		init_masked_lumped_mass(ndof, mass, obstacle_ndof);
		update_target(t); // initialize target_x_
	}

	BCPenaltyForm::BCPenaltyForm(const int ndof,
								 const std::vector<int> &boundary_nodes,
								 const StiffnessMatrix &mass,
								 const size_t obstacle_ndof,
								 const Eigen::MatrixXd &target_x)
		: boundary_nodes_(boundary_nodes),
		  local_boundary_(nullptr),
		  local_neumann_boundary_(nullptr),
		  n_boundary_samples_(0),
		  rhs_assembler_(nullptr),
		  is_time_dependent_(false),
		  target_x_(target_x)
	{
		init_masked_lumped_mass(ndof, mass, obstacle_ndof);
	}

	void BCPenaltyForm::init_masked_lumped_mass(
		const int ndof,
		const StiffnessMatrix &mass,
		const size_t obstacle_ndof)
	{
		std::vector<bool> is_boundary_dof(ndof, true);
		for (const auto bn : boundary_nodes_)
			is_boundary_dof[bn] = false;

		masked_lumped_mass_ = mass.size() == 0 ? polyfem::utils::sparse_identity(ndof, ndof) : polyfem::utils::lump_matrix(mass);
		assert(ndof == masked_lumped_mass_.rows() && ndof == masked_lumped_mass_.cols());

		// Give the collision obstacles a entry in the lumped mass matrix
		if (obstacle_ndof > 0)
		{
			const int n_fe_dof = ndof - obstacle_ndof;
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

		mask_.resize(masked_lumped_mass_.rows(), masked_lumped_mass_.cols());
		mask_.setIdentity();
		mask_.prune([&](const int &row, const int &col, const double &value) -> bool {
			assert(row == col); // matrix should be diagonal
			return !is_boundary_dof[row];
		});
	}

	double BCPenaltyForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd dist = x - target_x_;
		const double AL_penalty = 0.5 * dist.transpose() * masked_lumped_mass_ * dist;
		return AL_penalty;
	}

	void BCPenaltyForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = masked_lumped_mass_ * (x - target_x_);
	}

	void BCPenaltyForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = masked_lumped_mass_;
	}

	void BCPenaltyForm::update_quantities(const double t, const Eigen::VectorXd &)
	{
		if (is_time_dependent_)
			update_target(t);
	}

	void BCPenaltyForm::update_target(const double t)
	{
		assert(rhs_assembler_ != nullptr);
		assert(local_boundary_ != nullptr);
		assert(local_neumann_boundary_ != nullptr);
		target_x_.setZero(masked_lumped_mass_.rows(), 1);
		rhs_assembler_->set_bc(
			*local_boundary_, boundary_nodes_, n_boundary_samples_,
			*local_neumann_boundary_, target_x_, Eigen::MatrixXd(), t);
	}
} // namespace polyfem::solver

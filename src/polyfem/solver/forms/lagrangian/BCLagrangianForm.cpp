#include "BCLagrangianForm.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/assembler/PeriodicBoundary.hpp>

namespace polyfem::solver
{
	BCLagrangianForm::BCLagrangianForm(const int ndof,
									   const std::vector<int> &boundary_nodes,
									   const std::vector<mesh::LocalBoundary> &local_boundary,
									   const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
									   const int n_boundary_samples,
									   const StiffnessMatrix &mass,
									   const assembler::RhsAssembler &rhs_assembler,
									   const size_t obstacle_ndof,
									   const bool is_time_dependent,
									   const double t,
									   const std::shared_ptr<utils::PeriodicBoundary> &periodic_bc)
		: AugmentedLagrangianForm(periodic_bc ? periodic_bc->full_to_periodic(boundary_nodes) : boundary_nodes),
		  boundary_nodes_(boundary_nodes),
		  local_boundary_(&local_boundary),
		  local_neumann_boundary_(&local_neumann_boundary),
		  n_boundary_samples_(n_boundary_samples),
		  rhs_assembler_(&rhs_assembler),
		  is_time_dependent_(is_time_dependent)
	{
		init_masked_lumped_mass(ndof, mass, obstacle_ndof);
		update_target(t); // initialize target_x_
	}

	BCLagrangianForm::BCLagrangianForm(const int ndof,
									   const std::vector<int> &boundary_nodes,
									   const StiffnessMatrix &mass,
									   const size_t obstacle_ndof,
									   const Eigen::MatrixXd &target_x)
		: AugmentedLagrangianForm(boundary_nodes),
		  boundary_nodes_(boundary_nodes),
		  local_boundary_(nullptr),
		  local_neumann_boundary_(nullptr),
		  n_boundary_samples_(0),
		  rhs_assembler_(nullptr),
		  is_time_dependent_(false),
		  target_x_(target_x)
	{
		init_masked_lumped_mass(ndof, mass, obstacle_ndof);
	}

	void BCLagrangianForm::init_masked_lumped_mass(
		const int ndof,
		const StiffnessMatrix &mass,
		const size_t obstacle_ndof)
	{
		std::vector<bool> is_boundary_dof(ndof, true);
		for (const auto bn : boundary_nodes_)
			is_boundary_dof[bn] = false;

		masked_lumped_mass_sqrt_ = mass.size() == 0 ? polyfem::utils::sparse_identity(ndof, ndof) : polyfem::utils::lump_matrix(mass);
		{
			double min_diag = std::numeric_limits<double>::max();
			double max_diag = 0;
			for (int k = 0; k < masked_lumped_mass_sqrt_.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(masked_lumped_mass_sqrt_, k); it; ++it)
				{
					if (it.col() == it.row())
					{
						min_diag = std::min(min_diag, it.value());
						max_diag = std::max(max_diag, it.value());
					}
				}
			}
			if (max_diag <= 0 || min_diag <= 0 || min_diag / max_diag < 1e-16)
			{
				logger().warn("Lumped mass matrix ill-conditioned. Setting lumped mass matrix to identity.");
				masked_lumped_mass_sqrt_ = polyfem::utils::sparse_identity(ndof, ndof);
			}
		}

		assert(ndof == masked_lumped_mass_sqrt_.rows() && ndof == masked_lumped_mass_sqrt_.cols());

		// Give the collision obstacles a entry in the lumped mass matrix
		if (obstacle_ndof != 0)
		{
			const int n_fe_dof = ndof - obstacle_ndof;
			const double avg_mass = masked_lumped_mass_sqrt_.diagonal().head(n_fe_dof).mean();
			for (int i = n_fe_dof; i < ndof; ++i)
			{
				masked_lumped_mass_sqrt_.coeffRef(i, i) = avg_mass;
			}
		}

		// Remove non-boundary ndof from mass matrix
		masked_lumped_mass_sqrt_.prune([&](const int &row, const int &col, const double &value) -> bool {
			assert(row == col); // matrix should be diagonal
			return !is_boundary_dof[row];
		});
		masked_lumped_mass_ = masked_lumped_mass_sqrt_;
		mask_.resize(masked_lumped_mass_.rows(), masked_lumped_mass_.cols());
		mask_.setIdentity();
		mask_.prune([&](const int &row, const int &col, const double &value) -> bool {
			assert(row == col); // matrix should be diagonal
			return !is_boundary_dof[row];
		});

		std::vector<Eigen::Triplet<double>> tmp_triplets;
		tmp_triplets.reserve(masked_lumped_mass_sqrt_.nonZeros());
		for (int k = 0; k < masked_lumped_mass_sqrt_.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(masked_lumped_mass_sqrt_, k); it; ++it)
			{
				assert(it.col() == k);
				tmp_triplets.emplace_back(it.row(), it.col(), sqrt(it.value()));
			}
		}

		masked_lumped_mass_sqrt_.setFromTriplets(tmp_triplets.begin(), tmp_triplets.end());
		masked_lumped_mass_sqrt_.makeCompressed();

		lagr_mults_.resize(ndof);
		lagr_mults_.setZero();
	}

	double BCLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd dist = x - target_x_;
		const double L_penalty = -lagr_mults_.transpose() * masked_lumped_mass_sqrt_ * dist;
		const double A_penalty = 0.5 * dist.transpose() * masked_lumped_mass_ * dist;

		return L_penalty + k_al_ * A_penalty;
	}

	void BCLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = -(masked_lumped_mass_sqrt_ * lagr_mults_) + k_al_ * (masked_lumped_mass_ * (x - target_x_));
	}

	void BCLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = k_al_ * masked_lumped_mass_;
	}

	void BCLagrangianForm::update_quantities(const double t, const Eigen::VectorXd &)
	{
		if (is_time_dependent_)
			update_target(t);
	}

	double BCLagrangianForm::compute_error(const Eigen::VectorXd &x) const
	{
		return (this->target(x) - x).transpose() * this->mask() * (this->target(x) - x);
	}

	void BCLagrangianForm::update_target(const double t)
	{
		assert(rhs_assembler_ != nullptr);
		assert(local_boundary_ != nullptr);
		assert(local_neumann_boundary_ != nullptr);
		target_x_.setZero(masked_lumped_mass_sqrt_.rows(), 1);
		rhs_assembler_->set_bc(
			*local_boundary_, boundary_nodes_, n_boundary_samples_,
			*local_neumann_boundary_, target_x_, Eigen::MatrixXd(), t);
	}

	void BCLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
	{
		k_al_ = k_al;
		lagr_mults_ -= k_al_ * masked_lumped_mass_sqrt_ * (x - target_x_);
	}
} // namespace polyfem::solver

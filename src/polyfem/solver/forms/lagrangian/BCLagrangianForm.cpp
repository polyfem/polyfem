#include "BCLagrangianForm.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/assembler/PeriodicBoundary.hpp>
#include <igl/slice.h>

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
		: boundary_nodes_(boundary_nodes),
		  local_boundary_(&local_boundary),
		  local_neumann_boundary_(&local_neumann_boundary),
		  n_boundary_samples_(n_boundary_samples),
		  rhs_assembler_(&rhs_assembler),
		  is_time_dependent_(is_time_dependent),
		  n_dofs_(ndof)
	{
		init_masked_lumped_mass(mass, obstacle_ndof);
		update_target(t); // initialize b_
	}

	BCLagrangianForm::BCLagrangianForm(const int ndof,
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
		  n_dofs_(ndof)
	{
		init_masked_lumped_mass(mass, obstacle_ndof);

		b_ = target_x;
		igl::slice(b_, constraints_, 1, b_);
	}

	void BCLagrangianForm::init_masked_lumped_mass(
		const StiffnessMatrix &mass,
		const size_t obstacle_ndof)
	{
		std::vector<Eigen::Triplet<double>> A_triplets;

		constraints_.resize(boundary_nodes_.size());
		for (int i = 0; i < boundary_nodes_.size(); ++i)
		{
			const int bn = boundary_nodes_[i];

			constraints_[i] = bn;
			A_triplets.emplace_back(i, bn, 1.0);
		}
		A_.resize(boundary_nodes_.size(), n_dofs_);
		A_.setFromTriplets(A_triplets.begin(), A_triplets.end());
		A_.makeCompressed();

		masked_lumped_mass_ = mass.size() == 0 ? polyfem::utils::sparse_identity(n_dofs_, n_dofs_) : polyfem::utils::lump_matrix(mass);
		assert(n_dofs_ == masked_lumped_mass_.rows() && n_dofs_ == masked_lumped_mass_.cols());
		// Give the collision obstacles a entry in the lumped mass matrix
		if (obstacle_ndof != 0)
		{
			const int n_fe_dof = n_dofs_ - obstacle_ndof;
			const double avg_mass = masked_lumped_mass_.diagonal().head(n_fe_dof).mean();
			for (int i = n_fe_dof; i < n_dofs_; ++i)
			{
				masked_lumped_mass_.coeffRef(i, i) = avg_mass;
			}
		}
		igl::slice(masked_lumped_mass_, constraints_, 1, masked_lumped_mass_);
		igl::slice(masked_lumped_mass_, constraints_, 2, masked_lumped_mass_);
		assert(boundary_nodes_.size() == masked_lumped_mass_.rows() && boundary_nodes_.size() == masked_lumped_mass_.cols());

		masked_lumped_mass_sqrt_.resize(masked_lumped_mass_.rows(), masked_lumped_mass_.cols());
		std::vector<Eigen::Triplet<double>> tmp_triplets;
		tmp_triplets.reserve(masked_lumped_mass_.nonZeros());
		for (int k = 0; k < masked_lumped_mass_.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(masked_lumped_mass_, k); it; ++it)
			{
				assert(it.col() == k);
				tmp_triplets.emplace_back(it.row(), it.col(), sqrt(it.value()));
			}
		}

		masked_lumped_mass_sqrt_.setFromTriplets(tmp_triplets.begin(), tmp_triplets.end());
		masked_lumped_mass_sqrt_.makeCompressed();

		lagr_mults_.resize(boundary_nodes_.size());
		lagr_mults_.setZero();
	}

	double BCLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd dist = A_ * x - b_;
		const double L_penalty = -lagr_mults_.transpose() * masked_lumped_mass_sqrt_ * dist;
		const double A_penalty = 0.5 * dist.transpose() * masked_lumped_mass_ * dist;

		return L_penalty + k_al_ * A_penalty;
	}

	void BCLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = A_.transpose() * (-(masked_lumped_mass_sqrt_ * lagr_mults_) + k_al_ * (masked_lumped_mass_ * (A_ * x - b_)));
	}

	void BCLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = k_al_ * A_.transpose() * masked_lumped_mass_ * A_;
	}

	void BCLagrangianForm::update_quantities(const double t, const Eigen::VectorXd &)
	{
		if (is_time_dependent_)
			update_target(t);
	}

	double BCLagrangianForm::compute_error(const Eigen::VectorXd &x) const
	{
		// return (b_ - x).transpose() * A_ * (b_ - x);
		const Eigen::VectorXd res = A_ * x - b_;
		return res.squaredNorm();
	}

	void BCLagrangianForm::update_target(const double t)
	{
		assert(rhs_assembler_ != nullptr);
		assert(local_boundary_ != nullptr);
		assert(local_neumann_boundary_ != nullptr);
		b_.setZero(n_dofs_, 1);
		rhs_assembler_->set_bc(
			*local_boundary_, boundary_nodes_, n_boundary_samples_,
			*local_neumann_boundary_, b_, Eigen::MatrixXd(), t);

		b_ = igl::slice(b_, constraints_, 1);
	}

	void BCLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
	{
		k_al_ = k_al;
		lagr_mults_ -= k_al_ * masked_lumped_mass_sqrt_ * (A_ * x - b_);
	}
} // namespace polyfem::solver

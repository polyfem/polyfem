#include "BCLagrangianForm.hpp"

#include <polyfem/utils/Logger.hpp>
#include <igl/slice.h>

#ifdef POLYSOLVE_LARGE_INDEX
#include <unordered_map>
#include <cassert>
#endif

#ifdef POLYSOLVE_LARGE_INDEX
namespace
{
	// Custom slice implementation for large indices
	// Handles StiffnessMatrix with std::ptrdiff_t indices without converting to int
	template <typename DerivedR>
	void slice_stiffness_matrix(
		const polyfem::StiffnessMatrix &X,
		const Eigen::DenseBase<DerivedR> &R,
		const int dim,
		polyfem::StiffnessMatrix &Y)
	{
		using StorageIndex = std::ptrdiff_t;
		
		// Create a temporary result matrix to handle the case where X and Y are the same object
		polyfem::StiffnessMatrix result;
		
		if (dim == 1) // Slice rows: Y = X(R, :)
		{
			const StorageIndex new_rows = R.size();
			const StorageIndex new_cols = X.cols();
			
			std::vector<Eigen::Triplet<double, StorageIndex>> triplets;
			triplets.reserve(X.nonZeros()); // Upper bound estimate
			
			// Create mapping from old row indices to new row indices (supports duplicate indices)
			std::vector<std::vector<StorageIndex>> row_map;
			row_map.resize(X.rows());
			for (StorageIndex i = 0; i < new_rows; ++i)
			{
				const StorageIndex old_row = static_cast<StorageIndex>(R(i));
				assert(old_row >= 0 && old_row < X.rows() && "Row index out of bounds");
				row_map[old_row].push_back(i);
			}
			
			// Iterate through all non-zero elements (outerSize() = cols for ColMajor)
			for (StorageIndex k = 0; k < X.outerSize(); ++k)
			{
				for (typename polyfem::StiffnessMatrix::InnerIterator it(X, k); it; ++it)
				{
					for (const auto &new_row : row_map[it.row()])
					{
						triplets.emplace_back(new_row, it.col(), it.value());
					}
				}
			}
			
			result.resize(new_rows, new_cols);
			result.setFromTriplets(triplets.begin(), triplets.end());
		}
		else if (dim == 2) // Slice columns: Y = X(:, R)
		{
			const StorageIndex new_rows = X.rows();
			const StorageIndex new_cols = R.size();
			
			std::vector<Eigen::Triplet<double, StorageIndex>> triplets;
			triplets.reserve(X.nonZeros()); // Upper bound estimate
			
			// Create mapping from old column indices to new column indices (supports duplicate indices)
			std::vector<std::vector<StorageIndex>> col_map;
			col_map.resize(X.cols());
			for (StorageIndex i = 0; i < new_cols; ++i)
			{
				const StorageIndex old_col = static_cast<StorageIndex>(R(i));
				assert(old_col >= 0 && old_col < X.cols() && "Column index out of bounds");
				col_map[old_col].push_back(i);
			}
			
			// Iterate through all columns (outerSize() = cols for ColMajor)
			for (StorageIndex k = 0; k < X.outerSize(); ++k)
			{
				if (!col_map[k].empty()) // Only process columns that are selected
				{
					for (typename polyfem::StiffnessMatrix::InnerIterator it(X, k); it; ++it)
					{
						for (const auto &new_col : col_map[k])
						{
							triplets.emplace_back(it.row(), new_col, it.value());
						}
					}
				}
			}
			
			result.resize(new_rows, new_cols);
			result.setFromTriplets(triplets.begin(), triplets.end());
		}
		else
		{
			throw std::runtime_error("Invalid dimension for slice_stiffness_matrix (must be 1 or 2)");
		}
		
		result.makeCompressed();
		Y = std::move(result);
	}
}
#endif

namespace polyfem::solver
{
	BCLagrangianForm::BCLagrangianForm(const int ndof,
									   const std::vector<int> &boundary_nodes,
									   const std::vector<mesh::LocalBoundary> &local_boundary,
									   const std::vector<mesh::LocalBoundary> &local_neumann_boundary,
									   const QuadratureOrders &n_boundary_samples,
									   const StiffnessMatrix &mass,
									   const assembler::RhsAssembler &rhs_assembler,
									   const size_t obstacle_ndof,
									   const bool is_time_dependent,
									   const double t,
									   const BCLumpingMode lumping)
		: boundary_nodes_(boundary_nodes),
		  local_boundary_(&local_boundary),
		  local_neumann_boundary_(&local_neumann_boundary),
		  n_boundary_samples_(n_boundary_samples),
		  rhs_assembler_(&rhs_assembler),
		  is_time_dependent_(is_time_dependent),
		  n_dofs_(ndof)
	{
		lumping_ = lumping;
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
		  n_boundary_samples_({{0, 0}}),
		  rhs_assembler_(nullptr),
		  is_time_dependent_(false),
		  n_dofs_(ndof)
	{
		init_masked_lumped_mass(mass, obstacle_ndof);

		b_ = target_x;
		b_proj_ = b_;
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

		const auto is_ill_conditioned = [](const StiffnessMatrix &lumped) {
			double min_diag = std::numeric_limits<double>::max();
			double max_diag = 0;
			for (int k = 0; k < lumped.outerSize(); ++k)
			{
				for (StiffnessMatrix::InnerIterator it(lumped, k); it; ++it)
				{
					if (it.col() == it.row())
					{
						min_diag = std::min(min_diag, it.value());
						max_diag = std::max(max_diag, it.value());
					}
				}
			}
			return max_diag <= 0 || min_diag <= 0 || min_diag / max_diag < 1e-16;
		};

		if (mass.size() == 0)
			masked_lumped_mass_ = polyfem::utils::sparse_identity(n_dofs_, n_dofs_);
		else if (lumping_ == BCLumpingMode::HRZ)
			masked_lumped_mass_ = polyfem::utils::lump_matrix_hrz(mass);
		else
		{
			masked_lumped_mass_ = polyfem::utils::lump_matrix(mass);
			if (is_ill_conditioned(masked_lumped_mass_))
			{
				// row-sum lumping is invalid for bases of order >= 2 (rows of the
				// consistent mass can sum to zero or negative values)
				logger().warn("Row-sum lumped mass ill-conditioned (expected for order >= 2); using HRZ-style diagonal-scaling lumping for the BC penalty metric.");
				masked_lumped_mass_ = polyfem::utils::lump_matrix_hrz(mass);
			}
		}
		// last resort: only reject NONPOSITIVE metrics (a large diagonal spread
		// is physical grading, not damage), and keep mass units by scaling the
		// identity with the average nodal mass
		if (mass.size() != 0)
		{
			double min_diag = std::numeric_limits<double>::max();
			double diag_sum = 0;
			for (int k = 0; k < masked_lumped_mass_.outerSize(); ++k)
				for (StiffnessMatrix::InnerIterator it(masked_lumped_mass_, k); it; ++it)
					if (it.col() == it.row())
					{
						min_diag = std::min(min_diag, it.value());
						diag_sum += it.value();
					}
			if (min_diag <= 0)
			{
				const double avg_mass = diag_sum > 0 ? diag_sum / n_dofs_ : 1.0;
				logger().warn("Lumped mass matrix has nonpositive entries; using identity scaled by the average nodal mass ({}).", avg_mass);
				masked_lumped_mass_ = avg_mass * polyfem::utils::sparse_identity(n_dofs_, n_dofs_);
			}
		}

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

#ifdef POLYSOLVE_LARGE_INDEX
	slice_stiffness_matrix(masked_lumped_mass_, constraints_, 1, masked_lumped_mass_);
	slice_stiffness_matrix(masked_lumped_mass_, constraints_, 2, masked_lumped_mass_);
#else
	igl::slice(masked_lumped_mass_, constraints_, 1, masked_lumped_mass_);
	igl::slice(masked_lumped_mass_, constraints_, 2, masked_lumped_mass_);
#endif
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

		std::vector<bool> is_contraints(n_dofs_, false);
		for (int i = 0; i < boundary_nodes_.size(); ++i)
			is_contraints[boundary_nodes_[i]] = true;

		int index = 0;
		A_triplets.clear();
		not_constraints_.resize(n_dofs_ - boundary_nodes_.size());
		old_to_new_.assign(n_dofs_, -1);
		for (int i = 0; i < n_dofs_; ++i)
		{
			if (is_contraints[i])
				continue;

			A_triplets.emplace_back(i, index, 1.0);
			not_constraints_[index] = i;
			old_to_new_[i] = index;
			index++;
		}
		A_proj_.resize(n_dofs_, index);
		A_proj_.setFromTriplets(A_triplets.begin(), A_triplets.end());
		A_proj_.makeCompressed();

		lagr_mults_.resize(boundary_nodes_.size());
		lagr_mults_.setZero();
	}

	bool BCLagrangianForm::can_project() const { return true; }

	void BCLagrangianForm::project_gradient(Eigen::VectorXd &grad) const
	{
		// Assumes not_constraints_ is sorted
		for (int i = 0; i < not_constraints_.size(); ++i)
			grad[i] = grad[not_constraints_[i]];
		grad.conservativeResize(not_constraints_.size());
	}

	void BCLagrangianForm::project_diag(Eigen::VectorXd &diag) const
	{
		// Assumes not_constraints_ is sorted
		for (int i = 0; i < not_constraints_.size(); ++i)
			diag[i] = diag[not_constraints_[i]];
		diag.conservativeResize(not_constraints_.size());
	}

	void BCLagrangianForm::project_hessian(StiffnessMatrix &hessian) const
	{
		// Drop rows and columns whose indices are constrained DOFs in a single
		// linear scan over the ColMajor CSC storage, avoiding two back-to-back
		// igl::slice calls (each of which builds a triplet list and runs a
		// full setFromTriplets sort). On a 3D mat-twist run this cut the
		// call from ~63ms to ~5ms (~12x) and reduced NLProblem::hessian by
		// ~40%.
		assert(hessian.rows() == n_dofs_ && hessian.cols() == n_dofs_);
		hessian.makeCompressed();

		const int n_red = static_cast<int>(not_constraints_.size());

		// Pass 1: count total nnz of the reduced matrix.
		StiffnessMatrix::StorageIndex total_nnz = 0;
		for (int k = 0; k < hessian.outerSize(); ++k)
		{
			if (old_to_new_[k] < 0)
				continue;
			for (StiffnessMatrix::InnerIterator it(hessian, k); it; ++it)
			{
				if (old_to_new_[it.row()] >= 0)
					++total_nnz;
			}
		}

		// Pass 2: fill column-by-column in compressed CSC order. Because
		// not_constraints_ is sorted ascending, old_to_new_ is monotonic on
		// its non-negative entries, so the filtered rows come out ascending
		// within each column — which is what insertBack requires.
		StiffnessMatrix out(n_red, n_red);
		out.reserve(total_nnz);
		for (int k = 0; k < hessian.outerSize(); ++k)
		{
			const int new_col = old_to_new_[k];
			if (new_col < 0)
				continue;
			out.startVec(new_col);
			for (StiffnessMatrix::InnerIterator it(hessian, k); it; ++it)
			{
				const int new_row = old_to_new_[it.row()];
				if (new_row < 0)
					continue;
				out.insertBack(new_row, new_col) = it.value();
			}
		}
		out.finalize();

		hessian = std::move(out);
	}

	double BCLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd dist = A_ * x - b_;
		const double L_penalty = -lagr_mults_.transpose() * masked_lumped_mass_sqrt_ * dist;
		const double A_penalty = 0.5 * dist.transpose() * masked_lumped_mass_ * dist;

		return L_weight() * L_penalty + A_weight() * A_penalty;
	}

	void BCLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = L_weight() * A_.transpose() * (-(masked_lumped_mass_sqrt_ * lagr_mults_) + A_weight() * (masked_lumped_mass_ * (A_ * x - b_)));
	}

	void BCLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = A_weight() * A_.transpose() * masked_lumped_mass_ * A_;
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
		b_proj_ = b_;
		b_ = igl::slice(b_, constraints_, 1);
	}

	void BCLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
	{
		k_al_ = k_al;
		lagr_mults_ -= k_al * masked_lumped_mass_sqrt_ * (A_ * x - b_);
	}
} // namespace polyfem::solver

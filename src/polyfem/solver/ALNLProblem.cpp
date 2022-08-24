#include "ALNLProblem.hpp"

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/utils/Types.hpp>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>

#include <igl/write_triangle_mesh.h>

#include <unsupported/Eigen/SparseExtra>

namespace polyfem
{
	using namespace polysolve;
	using namespace assembler;
	using namespace utils;

	namespace solver
	{

		ALNLProblem::ALNLProblem(const State &state, const RhsAssembler &rhs_assembler, const double t, const double dhat, const double weight)
			: super(state, rhs_assembler, t, dhat, true), weight_(weight)
		{
			// stop_dist_ = 1e-2 * state.min_edge_length;

			const int ndof = state.n_bases * state.mesh->dimension();

			std::vector<bool> is_boundary_dof(ndof, true);
			for (const auto bn : state.boundary_nodes)
				is_boundary_dof[bn] = false;

			masked_lumped_mass_ = state.mass.size() == 0 ? sparse_identity(ndof, ndof) : lump_matrix(state.mass);
			assert(ndof == masked_lumped_mass_.rows() && ndof == masked_lumped_mass_.cols());
			// Remove non-boundary ndof from mass matrix
			masked_lumped_mass_.prune([&](const int &row, const int &col, const double &value) -> bool {
				assert(row == col); // matrix should be diagonal
				return !is_boundary_dof[row];
			});

			update_target(t);
		}

		void ALNLProblem::update_target(const double t)
		{
			target_x_.setZero(masked_lumped_mass_.rows(), 1);
			rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.n_boundary_samples(), state.local_neumann_boundary, target_x_, t);
		}

		void ALNLProblem::update_quantities(const double t, const TVector &x)
		{
			super::update_quantities(t, x);
			if (is_time_dependent)
			{
				update_target(t);
			}
		}

		double ALNLProblem::value(const TVector &x, const bool only_elastic)
		{
			const double val = super::value(x, only_elastic);

			// ₙ
			// ∑ ½ κ mₖ ‖ xₖ - x̂ₖ ‖² = ½ κ (x - x̂)ᵀ M (x - x̂)
			// ᵏ
			const TVector dist = x - target_x_;
			const double AL_penalty = weight_ / 2 * dist.transpose() * masked_lumped_mass_ * dist;

			// TODO: Implement Lagrangian potential if needed (i.e., penalty weight exceeds maximum)
			// ₙ    __
			// ∑ -⎷ mₖ λₖᵀ (xₖ - x̂ₖ) = -λᵀ M (x - x̂)
			// ᵏ

			logger().trace("AL_penalty={}", sqrt(AL_penalty));

			return val + AL_penalty;
		}

		void ALNLProblem::gradient_no_rhs(const TVector &x, Eigen::MatrixXd &grad, const bool only_elastic)
		{
			super::gradient_no_rhs(x, grad, only_elastic);

			grad += weight_ * masked_lumped_mass_ * (x - target_x_);

			// TODO: Implement Lagrangian potential if needed (i.e., penalty weight exceeds maximum)
		}

		void ALNLProblem::hessian_full(const TVector &x, THessian &hessian)
		{
			super::hessian_full(x, hessian);
			hessian += weight_ * masked_lumped_mass_;
			hessian.makeCompressed();
			// Hessian of Lagrangian potential is zero
		}

		bool ALNLProblem::stop(const TVector &x)
		{
			// TVector distv;
			// compute_distance(x, distv);
			// const double dist = distv.norm();

			return false; // dist < stop_dist_;
		}
	} // namespace solver
} // namespace polyfem

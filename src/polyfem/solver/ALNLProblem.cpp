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
			std::vector<Eigen::Triplet<double>> entries;

			// stop_dist_ = 1e-2 * state.min_edge_length;

			for (const auto bn : state.boundary_nodes)
				entries.emplace_back(bn, bn, 1.0);

			hessian_AL_.resize(state.n_bases * state.mesh->dimension(), state.n_bases * state.mesh->dimension());
			hessian_AL_.setFromTriplets(entries.begin(), entries.end());
			hessian_AL_.makeCompressed();

			update_target(t);

			std::vector<bool> mask(hessian_AL_.rows(), true);

			for (const auto bn : state.boundary_nodes)
				mask[bn] = false;

			for (int i = 0; i < mask.size(); ++i)
				if (mask[i])
					not_boundary_.push_back(i);
		}

		void ALNLProblem::update_target(const double t)
		{
			target_x_.setZero(hessian_AL_.rows(), 1);
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

		void ALNLProblem::compute_distance(const TVector &x, TVector &res)
		{
			res = x - target_x_;

			for (const auto bn : not_boundary_)
				res[bn] = 0;
		}

		double ALNLProblem::value(const TVector &x, const bool only_elastic)
		{
			const double val = super::value(x, only_elastic);

			// ₙ
			// ∑ ½ κ mₖ ‖ xₖ - x̂ₖ ‖² = ½ κ (xₖ - x̂ₖ)ᵀ M (xₖ - x̂ₖ)
			// ᵏ
			TVector distv;
			compute_distance(x, distv);
			// TODO: replace this with the actual mass matrix
			Eigen::SparseMatrix<double> M = sparse_identity(x.size(), x.size());
			const double AL_penalty = weight_ / 2 * distv.transpose() * M * distv;

			// TODO: Implement Lagrangian potential if needed (i.e., penalty weight exceeds maximum)
			// ₙ    __
			// ∑ -⎷ mₖ λₖᵀ (xₖ - x̂ₖ)
			// ᵏ

			logger().trace("AL_penalty={}", sqrt(AL_penalty));

			Eigen::MatrixXd ddd;
			compute_displaced_points(x, ddd);
			if (ddd.cols() == 2)
			{
				ddd.conservativeResize(ddd.rows(), 3);
				ddd.col(2).setZero();
			}

			return val + AL_penalty;
		}

		void ALNLProblem::gradient_no_rhs(const TVector &x, Eigen::MatrixXd &gradv, const bool only_elastic)
		{
			super::gradient_no_rhs(x, gradv, only_elastic);

			TVector grad_AL;
			compute_distance(x, grad_AL);
			//logger().trace("dist grad {}", tmp.norm());
			grad_AL *= weight_;

			gradv += grad_AL;
			// gradv = tmp;
		}

		void ALNLProblem::hessian_full(const TVector &x, THessian &hessian)
		{
			super::hessian_full(x, hessian);
			hessian += weight_ * hessian_AL_;
			hessian.makeCompressed();
		}

		bool ALNLProblem::stop(const TVector &x)
		{
			// TVector distv;
			// compute_distance(x, distv);
			// const double dist = distv.norm();

			return false; //dist < stop_dist_;
		}
	} // namespace solver
} // namespace polyfem

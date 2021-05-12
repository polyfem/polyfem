#include <polyfem/ALNLProblem.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/Types.hpp>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>

#include <igl/write_triangle_mesh.h>

#include <unsupported/Eigen/SparseExtra>

// #define USE_DIV_BARRIER_STIFFNESS

namespace polyfem
{
	using namespace polysolve;

	ALNLProblem::ALNLProblem(State &state, const RhsAssembler &rhs_assembler, const double t, const double dhat, const bool project_to_psd, const double weight)
		: super(state, rhs_assembler, t, dhat, project_to_psd, true), weight_(weight)
	{
		std::vector<Eigen::Triplet<double>> entries;

		// stop_dist_ = 1e-2 * state.min_edge_length;

		for (const auto bn : state.boundary_nodes)
			entries.emplace_back(bn, bn, 2 * weight_);

		hessian_.resize(state.n_bases * state.mesh->dimension(), state.n_bases * state.mesh->dimension());
		hessian_.setFromTriplets(entries.begin(), entries.end());
		hessian_.makeCompressed();

		displaced_.resize(hessian_.rows(), 1);
		displaced_.setZero();

		rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], state.local_neumann_boundary, displaced_, t);

		std::vector<bool> mask(hessian_.rows(), true);

		for (const auto bn : state.boundary_nodes)
			mask[bn] = false;

		for (int i = 0; i < mask.size(); ++i)
			if (mask[i])
				not_boundary_.push_back(i);
	}

	void ALNLProblem::update_quantities(const double t, const TVector &x)
	{
		super::update_quantities(t, x);
		if (is_time_dependent)
		{
			displaced_.resize(hessian_.rows(), 1);
			displaced_.setZero();

			rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.args["n_boundary_samples"], state.local_neumann_boundary, displaced_, t);
		}
	}

	void ALNLProblem::compute_distance(const TVector &x, TVector &res)
	{
		res = x - displaced_;

		for (const auto bn : not_boundary_)
			res[bn] = 0;
	}

	double ALNLProblem::value(const TVector &x)
	{
		const double val = super::value(x);
		TVector distv;
		compute_distance(x, distv);
		const double dist = distv.squaredNorm();

		logger().trace("dist {}", sqrt(dist));

		Eigen::MatrixXd ddd;
		compute_displaced_points(x, ddd);
		igl::write_triangle_mesh("step.obj", ddd, state.boundary_triangles);

#ifdef USE_DIV_BARRIER_STIFFNESS
		return val + weight_ * dist / _barrier_stiffness;
#else
		return val + weight_ * dist;
#endif
	}

	void ALNLProblem::gradient_no_rhs(const TVector &x, Eigen::MatrixXd &gradv)
	{
		TVector tmp;
		super::gradient_no_rhs(x, gradv);
		compute_distance(x, tmp);
		//logger().trace("dist grad {}", tmp.norm());
#ifdef USE_DIV_BARRIER_STIFFNESS
		tmp *= 2 * weight_ / _barrier_stiffness;
#else
		tmp *= 2 * weight_;
#endif

		gradv += tmp;
		// gradv = tmp;
	}

	void ALNLProblem::hessian_full(const TVector &x, THessian &hessian)
	{
		super::hessian_full(x, hessian);
#ifdef USE_DIV_BARRIER_STIFFNESS
		hessian += hessian_ / _barrier_stiffness;
#else
		hessian += hessian_;
#endif
		hessian.makeCompressed();
	}

	bool ALNLProblem::stop(const TVector &x)
	{
		// TVector distv;
		// compute_distance(x, distv);
		// const double dist = distv.norm();

		return false; //dist < stop_dist_;
	}
} // namespace polyfem

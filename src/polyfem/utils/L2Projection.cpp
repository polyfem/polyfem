#include "L2Projection.hpp"

#include <polyfem/solver/SparseNewtonDescentSolver.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/PardisoSupport>

namespace polyfem::utils
{
	using namespace polyfem::basis;
	using namespace polyfem::assembler;
	using namespace polyfem::solver;

	L2ProjectionOptimizationProblem::L2ProjectionOptimizationProblem(
		const State &state,
		const RhsAssembler &rhs_assembler,
		const THessian &M,
		const THessian &A,
		const TVector &u_prev,
		const double t,
		const double weight)
		: NLProblem(state, rhs_assembler, t, 1e-3), m_M(M), m_A(A), m_u_prev(u_prev), weight_(weight)
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

	void L2ProjectionOptimizationProblem::update_target(const double t)
	{
		target_x_.setZero(hessian_AL_.rows(), 1);
		rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.n_boundary_samples(), state.local_neumann_boundary, target_x_, t);
	}

	void L2ProjectionOptimizationProblem::compute_distance(const TVector &x, TVector &res)
	{
		res = x - target_x_;

		for (const auto bn : not_boundary_)
			res[bn] = 0;
	}

	double L2ProjectionOptimizationProblem::value(const TVector &_x)
	{
		TVector x;
		reduced_to_full(_x, x);

		const double val =
			double(0.5 * x.transpose() * m_M * x)
			- double(x.transpose() * m_A * m_u_prev);

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

		// Eigen::MatrixXd ddd;
		// compute_displaced_points(x, ddd);
		// if (ddd.cols() == 2)
		// {
		// 	ddd.conservativeResize(ddd.rows(), 3);
		// 	ddd.col(2).setZero();
		// }

		return val + AL_penalty;
	}

	void L2ProjectionOptimizationProblem::gradient(const TVector &_x, TVector &grad)
	{
		TVector x;
		reduced_to_full(_x, x);

		TVector grad_full = m_M * x - m_A * m_u_prev;

		TVector grad_AL;
		compute_distance(x, grad_AL);
		// logger().trace("dist grad {}", tmp.norm());
		grad_AL *= weight_;

		grad_full += grad_AL;
		full_to_reduced(grad_full, grad);
	}

	void L2ProjectionOptimizationProblem::hessian(const TVector &x, THessian &hessian)
	{
		full_hessian_to_reduced_hessian(m_M + weight_ * hessian_AL_, hessian);
	}

	const Eigen::MatrixXd &L2ProjectionOptimizationProblem::current_rhs()
	{
		// if (!rhs_computed)
		{
			rhs_assembler.compute_energy_grad(state.local_boundary, state.boundary_nodes, state.density, state.n_boundary_samples(), state.local_neumann_boundary, state.rhs, t, _current_rhs);
			// rhs_computed = true;
			assert(_current_rhs.size() == full_size);
			rhs_assembler.set_bc(std::vector<mesh::LocalBoundary>(), std::vector<int>(), state.n_boundary_samples(), state.local_neumann_boundary, _current_rhs, t);

			if (reduced_size != full_size)
			{
				// rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.n_boundary_samples(), state.local_neumann_boundary, _current_rhs, t);
				rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.n_boundary_samples(), std::vector<mesh::LocalBoundary>(), _current_rhs, t);
			}
		}

		return _current_rhs;
	}

	/// @brief Project the quantities in u on to the space spanned by mesh.bases.
	void L2_projection(
		const State &state,
		const RhsAssembler &rhs_assembler,
		const bool is_volume,
		const int size,
		const int n_from_basis,
		const std::vector<ElementBases> &from_bases,
		const std::vector<ElementBases> &from_gbases,
		const int n_to_basis,
		const std::vector<ElementBases> &to_bases,
		const std::vector<ElementBases> &to_gbases,
		const AssemblyValsCache &cache,
		const Eigen::MatrixXd &y,
		Eigen::MatrixXd &x,
		const bool lump_mass_matrix)
	{
		// solve M x = A y for x where M is the mass matrix and A is the cross mass matrix.
		Eigen::SparseMatrix<double> M, A;
		{
			MassMatrixAssembler assembler;
			Density no_density; // Density of one (i.e., no scaling of mass matrix)
			assembler.assemble(
				is_volume, size,
				n_to_basis, no_density, to_bases, to_gbases,
				cache, M);

			assembler.assemble_cross(
				is_volume, size,
				n_from_basis, from_bases, from_gbases,
				n_to_basis, to_bases, to_gbases,
				cache, A);

			// write_sparse_matrix_csv("M.csv", M);
			// write_sparse_matrix_csv("A.csv", A);
			// logger().critical("M =\n{}", Eigen::MatrixXd(M));
			// logger().critical("A =\n{}", Eigen::MatrixXd(A));
		}

		if (lump_mass_matrix)
		{
			M = lump_matrix(M);
		}

		// --------------------------------------------------------------------

		double al_weight = state.args["solver"]["augmented_lagrangian"]["initial_weight"];
		const double max_al_weight = state.args["solver"]["augmented_lagrangian"]["max_weight"];

		L2ProjectionOptimizationProblem problem(
			state, rhs_assembler, M, A, y.col(0),
			/*t=*/0, // TODO: use the correct time
			al_weight);

		Eigen::VectorXd sol = Eigen::VectorXd::Zero(M.rows());
		Eigen::VectorXd tmp_sol;

		problem.full_to_reduced(sol, tmp_sol);

		// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

		problem.line_search_begin(sol, tmp_sol);
		while (
			!std::isfinite(problem.value(tmp_sol))
			|| !problem.is_step_valid(sol, tmp_sol)
			|| !problem.is_step_collision_free(sol, tmp_sol))
		{
			problem.line_search_end();
			problem.set_weight(al_weight);
			logger().debug("Solving L2 Projection with weight {}", al_weight);

			cppoptlib::SparseNewtonDescentSolver<L2ProjectionOptimizationProblem> solver(
				state.args["solver"]["nonlinear"],
				state.args["solver"]["linear"]["solver"],
				state.args["solver"]["linear"]["precond"]);
			solver.setLineSearch(state.args["solver"]["nonlinear"]["line_search"]["method"]);
			problem.init(sol);
			tmp_sol = sol;
			solver.minimize(problem, tmp_sol);

			sol = tmp_sol;
			problem.full_to_reduced(sol, tmp_sol);
			problem.line_search_begin(sol, tmp_sol);

			al_weight *= 2;

			if (al_weight >= max_al_weight)
				log_and_throw_error(fmt::format("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight));
		}
		problem.line_search_end();

		// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

		cppoptlib::SparseNewtonDescentSolver<L2ProjectionOptimizationProblem> solver(
			state.args["solver"]["nonlinear"],
			state.args["solver"]["linear"]["solver"],
			state.args["solver"]["linear"]["precond"]);
		solver.setLineSearch(state.args["solver"]["nonlinear"]["line_search"]["method"]);
		problem.init(sol);
		solver.minimize(problem, tmp_sol);

		problem.reduced_to_full(tmp_sol, sol);

		// --------------------------------------------------------------------

		// Construct a linear solver for M
		Eigen::PardisoLU<decltype(M)> linear_solver;
		// linear_solver->setParameters(solver_params);
		const Eigen::SparseMatrix<double> &LHS = M; // NOTE: remove & if you want to have a more complicated LHS
		linear_solver.analyzePattern(LHS);
		linear_solver.factorize(LHS);

		const Eigen::MatrixXd rhs = A * y.rightCols(2);
		x.resize(rhs.rows(), y.cols());
		x.col(0) = sol;
		x.rightCols(2) = linear_solver.solve(rhs);
		double residual_error = (LHS * x.rightCols(2) - rhs).norm();
		logger().critical("residual error in L2 projection: {}", residual_error);
		assert(residual_error < 1e-12);
	}

} // namespace polyfem::utils
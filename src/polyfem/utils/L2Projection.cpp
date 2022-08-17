#include "L2Projection.hpp"

#include <polyfem/solver/ALNLProblem.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/PardisoSupport>

namespace polyfem::utils
{
	using namespace polyfem::basis;
	using namespace polyfem::assembler;
	using namespace polyfem::solver;

	class L2ProjectionOptimizationProblem : public cppoptlib::Problem<double>
	{
	public:
		typedef cppoptlib::Problem<double> super;
		using typename super::Scalar;
		using typename super::TVector;
		typedef StiffnessMatrix THessian;

		L2ProjectionOptimizationProblem(
			const State &state,
			const RhsAssembler &rhs_assembler,
			const Eigen::SparseMatrix<double> &M,
			const Eigen::SparseMatrix<double> &A,
			const Eigen::VectorXd &u_prev,
			const double t,
			const double weight);

		void init(const TVector &displacement) {}

		void update_target(const double t);

		virtual double value(const TVector &x) override;
		virtual void gradient(const TVector &x, TVector &grad) override;
		virtual void hessian(const TVector &x, THessian &hessian);

		// ====================================================================

		bool is_step_valid(const TVector &x0, const TVector &x1) { return true; }
		bool is_step_collision_free(const TVector &x0, const TVector &x1) { return true; }
		double max_step_size(const TVector &x0, const TVector &x1) { return 1.0; }
		bool is_intersection_free(const TVector &x) { return true; }

		void line_search_begin(const TVector &x0, const TVector &x1) {}
		void line_search_end() {}
		void post_step(const int iter_num, const TVector &x) {}

		virtual bool stop(const TVector &x) { return false; }

		double heuristic_max_step(const TVector &dx) { return 1.0; }

	protected:
		const State &state;
		const assembler::RhsAssembler &rhs_assembler;

		THessian m_M;
		THessian m_A;
		TVector m_u_prev;

		double weight_;
		THessian hessian_AL_;
		std::vector<int> not_boundary_;
		Eigen::MatrixXd target_x_; // actually a vector with the same size as x

		void compute_distance(const TVector &x, TVector &res);
	};

	L2ProjectionOptimizationProblem::L2ProjectionOptimizationProblem(
		const State &state,
		const RhsAssembler &rhs_assembler,
		const THessian &M,
		const THessian &A,
		const TVector &u_prev,
		const double t,
		const double weight)
		: state(state), rhs_assembler(rhs_assembler), m_M(M), m_A(A), m_u_prev(u_prev), weight_(weight)
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

	double L2ProjectionOptimizationProblem::value(const TVector &x)
	{
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

	void L2ProjectionOptimizationProblem::gradient(const TVector &x, TVector &grad)
	{
		grad = m_M * x - m_A * m_u_prev;

		TVector grad_AL;
		compute_distance(x, grad_AL);
		// logger().trace("dist grad {}", tmp.norm());
		grad_AL *= weight_;

		grad += grad_AL;
	}

	void L2ProjectionOptimizationProblem::hessian(const TVector &x, THessian &hessian)
	{
		hessian = m_M + weight_ * hessian_AL_;
		hessian.makeCompressed();
	}

	/// @brief Project the quantities in u on to the space spanned by mesh.bases.
	void L2_projection(
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

		// Construct a linear solver for M
		Eigen::PardisoLU<decltype(M)> solver;
		// linear_solver->setParameters(solver_params);
		const Eigen::SparseMatrix<double> &LHS = M; // NOTE: remove & if you want to have a more complicated LHS
		solver.analyzePattern(LHS);
		solver.factorize(LHS);

		const Eigen::MatrixXd rhs = A * y;
		x.resize(rhs.rows(), rhs.cols());
		x = solver.solve(rhs);
		double residual_error = (LHS * x - rhs).norm();
		logger().critical("residual error in L2 projection: {}", residual_error);
		assert(residual_error < 1e-12);
	}

} // namespace polyfem::utils
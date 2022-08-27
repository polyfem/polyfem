#pragma once

#include <polyfem/State.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/assembler/MassMatrixAssembler.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALNLProblem.hpp>

namespace polyfem::mesh
{
	using namespace polyfem::solver;
	using namespace polyfem::assembler;

	/// @brief Project the quantities in u on to the space spanned by mesh.bases.
	void L2_projection(
		const State &state,
		const polyfem::assembler::RhsAssembler &rhs_assembler,
		const bool is_volume,
		const int size,
		const int n_basis_a,
		const std::vector<polyfem::basis::ElementBases> &bases_a,
		const std::vector<polyfem::basis::ElementBases> &gbases_a,
		const int n_basis_b,
		const std::vector<polyfem::basis::ElementBases> &bases_b,
		const std::vector<polyfem::basis::ElementBases> &gbases_b,
		const polyfem::assembler::AssemblyValsCache &cache,
		const Eigen::MatrixXd &y,
		Eigen::MatrixXd &x,
		const bool lump_mass_matrix = false);

	class L2ProjectionOptimizationProblem : public NLProblem
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
		virtual double value(const TVector &x, const bool only_elastic) override { return value(x); }
		virtual void gradient(const TVector &x, TVector &grad) override;
		virtual void hessian(const TVector &x, THessian &hessian) override;

		virtual const Eigen::MatrixXd &current_rhs() override;

		// ====================================================================

		bool is_step_valid(const TVector &x0, const TVector &x1) override { return true; }
		bool is_step_collision_free(const TVector &x0, const TVector &x1) override { return true; }
		double max_step_size(const TVector &x0, const TVector &x1) override { return 1.0; }
		bool is_intersection_free(const TVector &x) override { return true; }

		void line_search_begin(const TVector &x0, const TVector &x1) override {}
		void line_search_end() override {}
		void post_step(const int iter_num, const TVector &x) override {}

		virtual bool stop(const TVector &x) override { return false; }

		double heuristic_max_step(const TVector &dx) override { return 1.0; }

		void set_weight(const double w) { weight_ = w; }

	protected:
		using NLProblem::rhs_assembler;
		using NLProblem::state;

		THessian m_M;
		THessian m_A;
		TVector m_u_prev;

		double weight_;
		THessian hessian_AL_;
		std::vector<int> not_boundary_;
		Eigen::MatrixXd target_x_; // actually a vector with the same size as x

		Eigen::MatrixXd _current_rhs;

		void compute_distance(const TVector &x, TVector &res);
	};

} // namespace polyfem::mesh
#include "NLProblem.hpp"

#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/utils/Logger.hpp>

#include <polysolve/linear/Solver.hpp>
#ifdef POLYSOLVE_WITH_SPQR
#include <Eigen/SPQRSupport>
#include <SuiteSparseQR.hpp>
#endif
#include <igl/cat.h>
#include <igl/Timer.h>

#include <set>

/*
m \frac{\partial^2 u}{\partial t^2} = \psi = \text{div}(\sigma[u])\newline
u^{t+1} = u(t+\Delta t)\approx u(t) + \Delta t \dot u + \frac{\Delta t^2} 2 \ddot u \newline
= u(t) + \Delta t \dot u + \frac{\Delta t^2}{2} \psi\newline
M u^{t+1}_h \approx M u^t_h + \Delta t M v^t_h + \frac{\Delta t^2} {2} A u^{t+1}_h \newline
%
M (u^{t+1}_h - (u^t_h + \Delta t v^t_h)) - \frac{\Delta t^2} {2} A u^{t+1}_h
*/
// mü = ψ = div(σ[u])
// uᵗ⁺¹ = u(t + Δt) ≈ u(t) + Δtu̇ + ½Δt²ü = u(t) + Δtu̇ + ½Δt²ψ
// Muₕᵗ⁺¹ ≈ Muₕᵗ + ΔtMvₕᵗ ½Δt²Auₕᵗ⁺¹
// Root-finding form:
// M(uₕᵗ⁺¹ - (uₕᵗ + Δtvₕᵗ)) - ½Δt²Auₕᵗ⁺¹ = 0

namespace polyfem::solver
{
	NLProblem::NLProblem(
		const int full_size,
		const std::vector<std::shared_ptr<Form>> &forms,
		const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &penalty_forms,
		const std::shared_ptr<polysolve::linear::Solver> &solver)
		: FullNLProblem(forms),
		  full_size_(full_size),
		  t_(0),
		  penalty_forms_(penalty_forms),
		  solver_(solver)
	{
		setup_constraints();
		use_reduced_size();
	}

	NLProblem::NLProblem(
		const int full_size,
		const std::shared_ptr<utils::PeriodicBoundary> &periodic_bc,
		const double t,
		const std::vector<std::shared_ptr<Form>> &forms,
		const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &penalty_forms,
		const std::shared_ptr<polysolve::linear::Solver> &solver)
		: FullNLProblem(forms),
		  full_size_(full_size),
		  t_(t),
		  penalty_forms_(penalty_forms),
		  solver_(solver)
	{
		if (periodic_bc != nullptr)
			throw std::runtime_error("To be fixed");

		setup_constraints();
		use_reduced_size();
	}

	void NLProblem::setup_constraints()
	{
		if (penalty_forms_.empty())
		{
			reduced_size_ = full_size_;
			return;
		}

		std::vector<Eigen::Triplet<double>> Ae;
		int index = 0;
		for (const auto &f : penalty_forms_)
		{
			const auto &tmp = f->constraint_matrix();
			for (int i = 0; i < tmp.outerSize(); i++)
			{
				for (typename StiffnessMatrix::InnerIterator it(tmp, i); it; ++it)
				{
					Ae.emplace_back(index + it.row(), it.col(), it.value());
				}
			}
			index += tmp.rows();
		}
		StiffnessMatrix A(index, full_size_);
		A.setFromTriplets(Ae.begin(), Ae.end());
		A.makeCompressed();

		const int constraint_size = A.rows();
		reduced_size_ = full_size_ - constraint_size;
		Eigen::SparseMatrix<double, Eigen::ColMajor, long> At = A.transpose();
		At.makeCompressed();

		logger().debug("Constraint size: {} x {}", A.rows(), A.cols());

#ifdef POLYSOLVE_WITH_SPQR
		igl::Timer timer;
		timer.start();

		cholmod_common cc;
		cholmod_l_start(&cc); // start CHOLMOD

		// const int ordering = 0; // all, except 3:given treated as 0:fixed
		const int ordering = SPQR_ORDERING_DEFAULT; // all, except 3:given treated as 0:fixed
		const double tol = SPQR_DEFAULT_TOL;
		SuiteSparse_long econ = At.rows();
		SuiteSparse_long *E; // permutation of 0:n-1, NULL if identity

		cholmod_sparse Ac, *Qc, *Rc;

		long *p = At.outerIndexPtr();

		Ac.nzmax = At.nonZeros();
		Ac.nrow = At.rows();
		Ac.ncol = At.cols();
		Ac.p = p;
		Ac.i = At.innerIndexPtr();
		Ac.x = At.valuePtr();
		Ac.z = 0;
		Ac.sorted = 1;
		Ac.packed = 1;
		Ac.nz = 0;
		Ac.dtype = 0;
		Ac.stype = -1;
		Ac.xtype = CHOLMOD_REAL;
		Ac.dtype = CHOLMOD_DOUBLE;
		Ac.stype = 0;
		Ac.itype = CHOLMOD_LONG;

		const auto rank = SuiteSparseQR<double>(ordering, tol, econ, &Ac,
												// outputs
												&Qc, &Rc, &E, &cc);

		if (!Rc)
			log_and_throw_error("Failed to factorize constraints matrix");

		const auto n = Rc->ncol;
		P_.resize(n);
		if (E)
		{
			for (long j = 0; j < n; j++)
				P_.indices()(j) = E[j];

			std::cout << "Asdasd" << std::endl;
		}
		else
			P_.setIdentity();

		if (Qc->stype != 0 || Qc->sorted != 1 || Qc->packed != 1 || Rc->stype != 0 || Rc->sorted != 1 || Rc->packed != 1)
			log_and_throw_error("Q and R must be unsymmetric sorted and packed");

		const StiffnessMatrix Q = Eigen::Map<Eigen::SparseMatrix<double, Eigen::ColMajor, long>>(
			Qc->nrow, Qc->ncol, Qc->nzmax,
			static_cast<long *>(Qc->p), static_cast<long *>(Qc->i), static_cast<double *>(Qc->x));

		const StiffnessMatrix R = Eigen::Map<Eigen::SparseMatrix<double, Eigen::ColMajor, long>>(
			Rc->nrow, Rc->ncol, Rc->nzmax,
			static_cast<long *>(Rc->p), static_cast<long *>(Rc->i), static_cast<double *>(Rc->x));

		cholmod_l_free_sparse(&Qc, &cc);
		cholmod_l_free_sparse(&Rc, &cc);
		std::free(E);
		cholmod_l_finish(&cc);

		timer.stop();
		logger().debug("QR took: {}", timer.getElapsedTime());
#else
		igl::Timer timer;
		timer.start();

		// Eigen::SparseQR<StiffnessMatrix, Eigen::NaturalOrdering<int>> QR(At);
		Eigen::SparseQR<StiffnessMatrix, Eigen::COLAMDOrdering<int>> QR(At);

		timer.stop();
		logger().debug("QR took: {}", timer.getElapsedTime());

		if (QR.info() != Eigen::Success)
			log_and_throw_error("Failed to factorize constraints matrix");

		timer.start();
		StiffnessMatrix Q;
		Q = QR.matrixQ();
		timer.stop();
		logger().debug("Computation of Q took: {}", timer.getElapsedTime());

		const Eigen::SparseMatrix<double, Eigen::RowMajor> R = QR.matrixR();

		P_ = QR.colsPermutation();
#endif

		timer.start();

		Q1_ = Q.leftCols(constraint_size);
		assert(Q1_.rows() == full_size_);
		assert(Q1_.cols() == constraint_size);

		Q2_ = Q.rightCols(reduced_size_);
		assert(Q2_.rows() == full_size_);
		assert(Q2_.cols() == reduced_size_);

		R1_ = R.topRows(constraint_size);
		assert(R1_.rows() == constraint_size);
		assert(R1_.cols() == constraint_size);

		assert((Q1_.transpose() * Q2_).norm() < 1e-10);

		timer.stop();
		logger().debug("Getting Q1 Q2, R1 took: {}", timer.getElapsedTime());

		timer.start();
		StiffnessMatrix Q2tQ2 = Q2_.transpose() * Q2_;
		solver_->analyze_pattern(Q2tQ2, Q2tQ2.rows());
		solver_->factorize(Q2tQ2);
		logger().debug("Factorization and computation of Q2tQ2 took: {}", timer.getElapsedTime());

#ifndef NDEBUG
		StiffnessMatrix test = R.bottomRows(reduced_size_);
		assert(test.nonZeros() == 0);
#endif

		assert((Q1_ * R1_ - At * P_).norm() < 1e-10);

		std::vector<std::shared_ptr<Form>> tmp;
		tmp.insert(tmp.end(), penalty_forms_.begin(), penalty_forms_.end());
		penalty_problem_ = std::make_shared<FullNLProblem>(tmp);

		update_constraint_values();
	}

	void NLProblem::update_constraint_values()
	{
		// x =  Q1 * R1^(-T) * P^T b  +  Q2 * y
		int index = 0;
		TVector constraint_values(Q1_.cols());
		for (const auto &f : penalty_forms_)
		{
			constraint_values.segment(index, f->constraint_value().rows()) = f->constraint_value();
			index += f->constraint_value().rows();
		}

		constraint_values = P_.transpose() * constraint_values;

		const Eigen::VectorXd sol = R1_.transpose().triangularView<Eigen::Lower>().solve(constraint_values);
		assert((R1_.transpose() * sol - constraint_values).norm() < 1e-10);

		Q1R1iTb_ = Q1_ * sol;
	}

	void NLProblem::init_lagging(const TVector &x)
	{
		FullNLProblem::init_lagging(reduced_to_full(x));

		if (full_size() == current_size())
			penalty_problem_->init_lagging(x);
	}

	void NLProblem::update_lagging(const TVector &x, const int iter_num)
	{
		FullNLProblem::update_lagging(reduced_to_full(x), iter_num);

		if (full_size() == current_size())
			penalty_problem_->update_lagging(x, iter_num);
	}

	void NLProblem::update_quantities(const double t, const TVector &x)
	{
		assert(x.size() == full_size_);
		t_ = t;
		const TVector full = reduced_to_full(x);
		for (auto &f : forms_)
			f->update_quantities(t, full);

		for (auto &f : penalty_forms_)
			f->update_quantities(t, x);

		update_constraint_values();
	}

	void NLProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		FullNLProblem::line_search_begin(reduced_to_full(x0), reduced_to_full(x1));

		if (full_size() == current_size())
			penalty_problem_->line_search_begin(x0, x1);
	}

	double NLProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		double max_step = FullNLProblem::max_step_size(reduced_to_full(x0), reduced_to_full(x1));

		if (full_size() == current_size())
			max_step = std::min(max_step, penalty_problem_->max_step_size(x0, x1));

		return max_step;
	}

	bool NLProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		bool valid = FullNLProblem::is_step_valid(reduced_to_full(x0), reduced_to_full(x1));

		if (valid && full_size() == current_size())
			valid = penalty_problem_->is_step_valid(x0, x1);

		return valid;
	}

	bool NLProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
	{
		bool free = FullNLProblem::is_step_collision_free(reduced_to_full(x0), reduced_to_full(x1));

		if (free && full_size() == current_size())
			free = penalty_problem_->is_step_collision_free(x0, x1);

		return free;
	}

	double NLProblem::value(const TVector &x)
	{
		// TODO: removed fearure const bool only_elastic
		double res = FullNLProblem::value(reduced_to_full(x));

		if (full_size() == current_size())
		{
			res += penalty_problem_->value(x);
		}

		return res;
	}

	void NLProblem::gradient(const TVector &x, TVector &grad)
	{
		FullNLProblem::gradient(reduced_to_full(x), grad);

		if (full_size() != current_size())
		{
			grad = Q2_.transpose() * grad;
		}
		else
		{
			TVector tmp;
			penalty_problem_->gradient(x, tmp);
			grad += tmp;
		}
	}

	void NLProblem::hessian(const TVector &x, THessian &hessian)
	{
		FullNLProblem::hessian(reduced_to_full(x), hessian);

		if (full_size() != current_size())
		{
			hessian = Q2_.transpose() * hessian * Q2_;

			// remove numerical zeros
			hessian.prune([](const Eigen::Index &row, const Eigen::Index &col, const Scalar &value) {
				return std::abs(value) > 1e-10;
			});
		}
		else
		{
			THessian tmp;
			penalty_problem_->hessian(x, tmp);
			hessian += tmp;
		}
	}

	void NLProblem::solution_changed(const TVector &newX)
	{
		FullNLProblem::solution_changed(reduced_to_full(newX));

		if (full_size() == current_size())
			penalty_problem_->solution_changed(newX);
	}

	void NLProblem::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		FullNLProblem::post_step(polysolve::nonlinear::PostStepData(data.iter_num, data.solver_info, reduced_to_full(data.x), reduced_to_full(data.grad)));

		if (full_size() == current_size())
			penalty_problem_->post_step(data);

		// TODO: add me back
		// if (state_.args["output"]["advanced"]["save_nl_solve_sequence"])
		// {
		// 	const Eigen::MatrixXd displacements = utils::unflatten(reduced_to_full(x), state_.mesh->dimension());
		// 	io::OBJWriter::write(
		// 		state_.resolve_output_path(fmt::format("nonlinear_solve_iter{:03d}.obj", iter_num)),
		// 		state_.collision_mesh.displace_vertices(displacements),
		// 		state_.collision_mesh.edges(), state_.collision_mesh.faces());
		// }
	}

	NLProblem::TVector NLProblem::full_to_reduced(const TVector &full) const
	{
		// Reduced is already at the full size
		if (full_size() == current_size() || full.size() == current_size())
		{
			return full;
		}

		TVector reduced(reduced_size());
		const TVector k = full - Q1R1iTb_;
		const TVector rhs = Q2_.transpose() * k;
		solver_->solve(rhs, reduced);

#ifndef NDEBUG
		StiffnessMatrix Q2tQ2 = Q2_.transpose() * Q2_;
		assert((Q2tQ2 * reduced - rhs).norm() < 1e-10);
		// std::cout << "err " << (Q2_ * reduced - k).norm() << std::endl;
#endif

		return reduced;
	}

	NLProblem::TVector NLProblem::reduced_to_full(const TVector &reduced) const
	{
		// Full is already at the reduced size
		if (full_size() == current_size() || full_size() == reduced.size())
		{
			return reduced;
		}

		// x =  Q1 * R1^(-T) * P^T b  +  Q2 * y

		const TVector full = Q1R1iTb_ + Q2_ * reduced;

		// std::cout << "At Q2 y" << (At.transpose() * Q2_ * reduced).norm() << std::endl;
		// std::cout << "At Ai b" << (At.transpose() * Q1R1iTb_ - constraint_values_).norm() << std::endl;

		// std::cout << "At Ai b\n"
		// 		  << (At.transpose() * Q1R1iTb_ - constraint_values_) << std::endl;

#ifndef NDEBUG
		for (const auto &f : penalty_forms_)
		{
			// std::cout << f->compute_error(full) << std::endl;
			assert(f->compute_error(full) < 1e-10);
		}
#endif

		return full;
	}

} // namespace polyfem::solver

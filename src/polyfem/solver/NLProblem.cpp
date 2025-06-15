#include "NLProblem.hpp"

#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/utils/Logger.hpp>

#include <polysolve/linear/Solver.hpp>
#ifdef POLYSOLVE_WITH_SPQR
#include <Eigen/SPQRSupport>
#include <SuiteSparseQR.hpp>
#endif

#include <armadillo>

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

	namespace
	{
#ifdef POLYSOLVE_WITH_SPQR
		void fill_cholmod(Eigen::SparseMatrix<double, Eigen::ColMajor, long> &mat, cholmod_sparse &cmat)
		{
			long *p = mat.outerIndexPtr();

			cmat.nzmax = mat.nonZeros();
			cmat.nrow = mat.rows();
			cmat.ncol = mat.cols();
			cmat.p = p;
			cmat.i = mat.innerIndexPtr();
			cmat.x = mat.valuePtr();
			cmat.z = 0;
			cmat.sorted = 1;
			cmat.packed = 1;
			cmat.nz = 0;
			cmat.dtype = 0;
			cmat.stype = -1;
			cmat.xtype = CHOLMOD_REAL;
			cmat.dtype = CHOLMOD_DOUBLE;
			cmat.stype = 0;
			cmat.itype = CHOLMOD_LONG;
		}
#endif
		arma::sp_mat fill_arma(const StiffnessMatrix &mat)
		{
			std::vector<unsigned long long> rowind_vect(mat.innerIndexPtr(), mat.innerIndexPtr() + mat.nonZeros());
			std::vector<unsigned long long> colptr_vect(mat.outerIndexPtr(), mat.outerIndexPtr() + mat.outerSize() + 1);
			std::vector<double> values_vect(mat.valuePtr(), mat.valuePtr() + mat.nonZeros());

			arma::dvec values(values_vect.data(), values_vect.size(), false);
			arma::uvec rowind(rowind_vect.data(), rowind_vect.size(), false);
			arma::uvec colptr(colptr_vect.data(), colptr_vect.size(), false);

			arma::sp_mat amat(rowind, colptr, values, mat.rows(), mat.cols(), false);

			return amat;
		}

		StiffnessMatrix fill_eigen(const arma::sp_mat &mat)
		{
			// convert to eigen sparse
			std::vector<long> outerIndexPtr(mat.row_indices, mat.row_indices + mat.n_nonzero);
			std::vector<long> innerIndexPtr(mat.col_ptrs, mat.col_ptrs + mat.n_cols + 1);

			const StiffnessMatrix out = Eigen::Map<const Eigen::SparseMatrix<double, Eigen::ColMajor, long>>(
				mat.n_rows, mat.n_cols, mat.n_nonzero, innerIndexPtr.data(), outerIndexPtr.data(), mat.values);
			return out;
		}

	} // namespace
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
		setup_constraints();
		use_reduced_size();
	}

	double NLProblem::normalize_forms()
	{
		return 1;

		// double total_weight = 0;
		// for (const auto &f : forms_)
		// 	total_weight += f->weight();
		// if (full_size() == current_size())
		// {
		// 	for (const auto &f : penalty_forms_)
		// 		total_weight += f->weight() * f->lagrangian_weight();
		// }

		// logger().debug("Normalizing forms with scale: {}", total_weight);

		// for (auto &f : forms_)
		// 	f->set_scale(total_weight);
		// for (auto &f : penalty_forms_)
		// 	f->set_scale(total_weight);

		// return total_weight;
	}

	void NLProblem::setup_constraints()
	{
		if (penalty_forms_.empty())
		{
			reduced_size_ = full_size_;
			return;
		}
		igl::Timer timer;

		if (penalty_forms_.size() == 1 && penalty_forms_.front()->has_projection())
		{
			Q2_ = penalty_forms_.front()->constraint_projection_matrix();
			Q2t_ = Q2_.transpose();

			reduced_size_ = Q2_.cols();
			num_penalty_constraints_ = full_size_ - reduced_size_;

			timer.start();
			StiffnessMatrix Q2tQ2 = Q2t_ * Q2_;
			solver_->analyze_pattern(Q2tQ2, Q2tQ2.rows());
			solver_->factorize(Q2tQ2);
			timer.stop();
			logger().debug("Factorization and computation of Q2tQ2 took: {}", timer.getElapsedTime());

			std::vector<std::shared_ptr<Form>> tmp;
			tmp.insert(tmp.end(), penalty_forms_.begin(), penalty_forms_.end());
			penalty_problem_ = std::make_shared<FullNLProblem>(tmp);

			update_constraint_values();

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

		int constraint_size = A.rows();
		num_penalty_constraints_ = A.rows();
		Eigen::SparseMatrix<double, Eigen::ColMajor, long> At = A.transpose();
		At.makeCompressed();

		logger().debug("Constraint size: {} x {}", A.rows(), A.cols());

#ifdef POLYSOLVE_WITH_SPQR
		timer.start();

		cholmod_common cc;
		cholmod_l_start(&cc); // start CHOLMOD

		// const int ordering = 0; // all, except 3:given treated as 0:fixed
		const int ordering = SPQR_ORDERING_DEFAULT; // all, except 3:given treated as 0:fixed
		const double tol = SPQR_DEFAULT_TOL;
		SuiteSparse_long econ = At.rows();
		SuiteSparse_long *E; // permutation of 0:n-1, NULL if identity

		cholmod_sparse Ac, *Qc, *Rc;

		fill_cholmod(At, Ac);

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

		for (; constraint_size >= 0; --constraint_size)
		{
			const StiffnessMatrix tmp = R.row(constraint_size);
			if (tmp.nonZeros() != 0)
			{
				constraint_size++;
				break;
			}
		}
		if (constraint_size != num_penalty_constraints_)
			logger().warn("Matrix A is not full rank, constraint size: {} instead of {}", constraint_size, num_penalty_constraints_);

		reduced_size_ = full_size_ - constraint_size;

		timer.start();

		Q1_ = Q.leftCols(constraint_size);
		assert(Q1_.rows() == full_size_);
		assert(Q1_.cols() == constraint_size);

		Q2_ = Q.rightCols(reduced_size_);
		Q2t_ = Q2_.transpose();

		assert(Q2_.rows() == full_size_);
		assert(Q2_.cols() == reduced_size_);

		R1_ = R.topRows(constraint_size);
		assert(R1_.rows() == constraint_size);
		assert(R1_.cols() == num_penalty_constraints_);

		assert((Q1_.transpose() * Q2_).norm() < 1e-10);

		timer.stop();
		logger().debug("Getting Q1 Q2, R1 took: {}", timer.getElapsedTime());

		timer.start();

		// arma::sp_mat q2a = fill_arma(Q2_);
		// arma::sp_mat q2tq2 = q2a.t() * q2a;
		// const StiffnessMatrix Q2tQ2 = fill_eigen(q2tq2);
		StiffnessMatrix Q2tQ2 = Q2t_ * Q2_;
		timer.stop();
		logger().debug("Getting Q2'*Q2, took: {}", timer.getElapsedTime());

		timer.start();
		solver_->analyze_pattern(Q2tQ2, Q2tQ2.rows());
		solver_->factorize(Q2tQ2);
		timer.stop();
		logger().debug("Factorization of Q2'*Q2 took: {}", timer.getElapsedTime());

#ifndef NDEBUG
		StiffnessMatrix test = R.bottomRows(reduced_size_);
		assert(test.nonZeros() == 0);

		StiffnessMatrix test1 = R1_.row(R1_.rows() - 1);
		assert(test1.nonZeros() != 0);
#endif

		// assert((Q1_ * R1_ - At * P_).norm() < 1e-10);

		std::vector<std::shared_ptr<Form>> tmp;
		tmp.insert(tmp.end(), penalty_forms_.begin(), penalty_forms_.end());
		penalty_problem_ = std::make_shared<FullNLProblem>(tmp);

		update_constraint_values();
	}

	void NLProblem::update_constraint_values()
	{
		if (penalty_forms_.size() == 1 && penalty_forms_.front()->has_projection())
		{
			Q1R1iTb_ = penalty_forms_.front()->constraint_projection_vector();
			return;
		}

		igl::Timer timer;
		timer.start();
		// x =  Q1 * R1^(-T) * P^T b  +  Q2 * y
		int index = 0;
		TVector constraint_values(num_penalty_constraints_);
		for (const auto &f : penalty_forms_)
		{
			constraint_values.segment(index, f->constraint_value().rows()) = f->constraint_value();
			index += f->constraint_value().rows();
		}
		constraint_values = P_.transpose() * constraint_values;

		Eigen::VectorXd sol;

		if (R1_.rows() == R1_.cols())
		{
			sol = R1_.transpose().triangularView<Eigen::Lower>().solve(constraint_values);
		}
		else
		{

#ifdef POLYSOLVE_WITH_SPQR
			Eigen::SparseMatrix<double, Eigen::ColMajor, long> R1t = R1_.transpose();
			cholmod_common cc;
			cholmod_l_start(&cc); // start CHOLMOD
			cholmod_sparse R1tc;
			fill_cholmod(R1t, R1tc);

			cholmod_dense b;
			b.nrow = constraint_values.size();
			b.ncol = 1;
			b.nzmax = constraint_values.size();
			b.d = constraint_values.size();
			b.x = constraint_values.data();
			b.z = 0;
			b.xtype = CHOLMOD_REAL;
			b.dtype = 0;

			const int ordering = SPQR_ORDERING_DEFAULT; // all, except 3:given treated as 0:fixed
			const double tol = SPQR_DEFAULT_TOL;

			cholmod_dense *solc = SuiteSparseQR<double>(ordering, tol, &R1tc, &b, &cc);

			sol = Eigen::Map<Eigen::VectorXd>(static_cast<double *>(solc->x), solc->nrow);

			cholmod_l_free_dense(&solc, &cc);
			cholmod_l_finish(&cc);
#else
			Eigen::SparseQR<StiffnessMatrix, Eigen::COLAMDOrdering<int>> solver;
			solver.compute(R1_.transpose());
			if (solver.info() != Eigen::Success)
			{
				log_and_throw_error("Failed to factorize R1^T");
			}
			sol = solver.solve(constraint_values);
#endif
		}

		assert((R1_.transpose() * sol - constraint_values).norm() < 1e-10);

		Q1R1iTb_ = Q1_ * sol;

		timer.stop();
		logger().debug("Computing Q1R1iTb took: {}", timer.getElapsedTime());
	}

	void NLProblem::init_lagging(const TVector &x)
	{
		FullNLProblem::init_lagging(reduced_to_full(x));

		if (penalty_problem_ && full_size() == current_size())
			penalty_problem_->init_lagging(x);
	}

	void NLProblem::update_lagging(const TVector &x, const int iter_num)
	{
		FullNLProblem::update_lagging(reduced_to_full(x), iter_num);

		if (penalty_problem_ && full_size() == current_size())
			penalty_problem_->update_lagging(x, iter_num);
	}

	void NLProblem::update_quantities(const double t, const TVector &x)
	{
		t_ = t;
		const TVector full = reduced_to_full(x);
		assert(full.size() == full_size_);
		for (auto &f : forms_)
			f->update_quantities(t, full);

		for (auto &f : penalty_forms_)
			f->update_quantities(t, x);

		update_constraint_values();
	}

	void NLProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		FullNLProblem::line_search_begin(reduced_to_full(x0), reduced_to_full(x1));

		if (penalty_problem_ && full_size() == current_size())
			penalty_problem_->line_search_begin(x0, x1);
	}

	double NLProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		double max_step = FullNLProblem::max_step_size(reduced_to_full(x0), reduced_to_full(x1));

		if (penalty_problem_ && full_size() == current_size())
			max_step = std::min(max_step, penalty_problem_->max_step_size(x0, x1));

		return max_step;
	}

	bool NLProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		bool valid = FullNLProblem::is_step_valid(reduced_to_full(x0), reduced_to_full(x1));

		if (penalty_problem_ && valid && full_size() == current_size())
			valid = penalty_problem_->is_step_valid(x0, x1);

		return valid;
	}

	bool NLProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
	{
		bool free = FullNLProblem::is_step_collision_free(reduced_to_full(x0), reduced_to_full(x1));

		if (penalty_problem_ && free && full_size() == current_size())
			free = penalty_problem_->is_step_collision_free(x0, x1);

		return free;
	}

	double NLProblem::value(const TVector &x)
	{
		// TODO: removed fearure const bool only_elastic
		double res = FullNLProblem::value(reduced_to_full(x));

		if (penalty_problem_ && full_size() == current_size())
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
			if (penalty_forms_.size() == 1 && penalty_forms_.front()->can_project())
				penalty_forms_.front()->project_gradient(grad);
			else
				grad = Q2t_ * grad;
		}
		else if (penalty_problem_)
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
			full_hessian_to_reduced_hessian(hessian);
		}
		else if (penalty_problem_)
		{
			THessian tmp;
			penalty_problem_->hessian(x, tmp);
			hessian += tmp;
		}
	}

	void NLProblem::solution_changed(const TVector &newX)
	{
		FullNLProblem::solution_changed(reduced_to_full(newX));

		if (penalty_problem_ && full_size() == current_size())
			penalty_problem_->solution_changed(newX);
	}

	void NLProblem::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		FullNLProblem::post_step(polysolve::nonlinear::PostStepData(data.iter_num, data.solver_info, reduced_to_full(data.x), reduced_to_full(data.grad)));

		if (penalty_problem_ && full_size() == current_size())
			penalty_problem_->post_step(data);

		// TODO: add me back
		static int nsolves = 0;
		if (data.iter_num == 0)
			nsolves++;
		// if (state && state->args["output"]["advanced"]["save_nl_solve_sequence"])
		// {
		// 	const Eigen::MatrixXd displacements = utils::unflatten(reduced_to_full(data.x), state->mesh->dimension());
		// 	io::OBJWriter::write(
		// 		state->resolve_output_path(fmt::format("nonlinear_solve{:04d}_iter{:04d}.obj", nsolves, data.iter_num)),
		// 		state->collision_mesh.displace_vertices(displacements),
		// 		state->collision_mesh.edges(), state->collision_mesh.faces());
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
		const TVector rhs = Q2t_ * k;
		solver_->solve(rhs, reduced);

#ifndef NDEBUG
		StiffnessMatrix Q2tQ2 = Q2t_ * Q2_;
		// std::cout << "err " << (Q2tQ2 * reduced - rhs).norm() << std::endl;
		assert((Q2tQ2 * reduced - rhs).norm() < 1e-8);
#endif

		return reduced;
	}

	NLProblem::TVector NLProblem::full_to_reduced_grad(const TVector &full) const
	{
		TVector grad = full;
		if (penalty_forms_.size() == 1 && penalty_forms_.front()->can_project())
			penalty_forms_.front()->project_gradient(grad);
		else
			grad = Q2t_ * grad;

		return grad;
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

#ifndef NDEBUG
		for (const auto &f : penalty_forms_)
		{
			// std::cout << f->compute_error(full) << std::endl;
			assert(f->compute_error(full) < 1e-8);
		}
#endif

		return full;
	}

	void NLProblem::full_hessian_to_reduced_hessian(StiffnessMatrix &hessian) const
	{
		if (penalty_forms_.size() == 1 && penalty_forms_.front()->can_project())
			penalty_forms_.front()->project_hessian(hessian);
		else
		{
			// arma::sp_mat q2a = fill_arma(Q2_);
			// arma::sp_mat ha = fill_arma(hessian);
			// arma::sp_mat q2thq2 = q2a.t() * ha * q2a;
			// hessian = fill_eigen(q2thq2);
			hessian = Q2t_ * hessian * Q2_;
			// remove numerical zeros
			hessian.prune([](const Eigen::Index &row, const Eigen::Index &col, const Scalar &value) {
				return std::abs(value) > 1e-10;
			});
		}
	}
} // namespace polyfem::solver

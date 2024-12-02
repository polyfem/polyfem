#include "NLProblem.hpp"

#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <polyfem/utils/Logger.hpp>

#include <polysolve/linear/Solver.hpp>

#include <igl/cat.h>

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
		StiffnessMatrix At = A.transpose();

		Eigen::SparseQR<StiffnessMatrix, Eigen::COLAMDOrdering<int>> QR(At);

		if (QR.info() != Eigen::Success)
			log_and_throw_error("Failed to factorize constraints matrix");

		StiffnessMatrix Q;
		Q = QR.matrixQ();

		const Eigen::SparseMatrix<double, Eigen::RowMajor> R = QR.matrixR();

		Q1_ = Q.leftCols(constraint_size);
		assert(Q1_.rows() == full_size_);
		assert(Q1_.cols() == constraint_size);

		Q2_ = Q.rightCols(reduced_size_);
		assert(Q2_.rows() == full_size_);
		assert(Q2_.cols() == reduced_size_);

		R1_ = R.topRows(constraint_size);
		assert(R1_.rows() == constraint_size);
		assert(R1_.cols() == constraint_size);

		StiffnessMatrix Q2tQ2 = Q2_.transpose() * Q2_;
		solver_->analyze_pattern(Q2tQ2, Q2tQ2.rows());
		solver_->factorize(Q2tQ2);

#ifndef NDEBUG
		StiffnessMatrix test = R.bottomRows(reduced_size_);
		assert(test.nonZeros() == 0);
#endif

		update_constraint_values();
	}

	void NLProblem::update_constraint_values()
	{
		int index = 0;
		constraint_values_.resize(Q1_.cols());
		for (const auto &f : penalty_forms_)
		{
			constraint_values_.segment(index, f->constraint_value().rows()) = f->constraint_value();
			index += f->constraint_value().rows();
		}

		// std::cout << constraint_values_ << std::endl;
		// std::cout << R1_.transpose() << std::endl;
		const Eigen::VectorXd sol = R1_.transpose().triangularView<Eigen::Lower>().solve(constraint_values_);
		// std::cout << (R1_.transpose() * sol - constraint_values_).norm() << std::endl;
		Q1R1iTb_ = Q1_ * sol;
	}

	void NLProblem::init_lagging(const TVector &x)
	{
		FullNLProblem::init_lagging(reduced_to_full(x));
	}

	void NLProblem::update_lagging(const TVector &x, const int iter_num)
	{
		FullNLProblem::update_lagging(reduced_to_full(x), iter_num);
	}

	void NLProblem::update_quantities(const double t, const TVector &x)
	{
		t_ = t;
		const TVector full = reduced_to_full(x);
		for (auto &f : forms_)
			f->update_quantities(t, full);

		update_constraint_values();
	}

	void NLProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		FullNLProblem::line_search_begin(reduced_to_full(x0), reduced_to_full(x1));
	}

	double NLProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		return FullNLProblem::max_step_size(reduced_to_full(x0), reduced_to_full(x1));
	}

	bool NLProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		return FullNLProblem::is_step_valid(reduced_to_full(x0), reduced_to_full(x1));
	}

	bool NLProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
	{
		return FullNLProblem::is_step_collision_free(reduced_to_full(x0), reduced_to_full(x1));
	}

	double NLProblem::value(const TVector &x)
	{
		// TODO: removed fearure const bool only_elastic
		double res = FullNLProblem::value(reduced_to_full(x));

		if (full_size() == current_size())
		{
			for (const auto &f : penalty_forms_)
			{
				if (!f->enabled())
					continue;
				res += f->value(x);
			}
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

			for (const auto &f : penalty_forms_)
			{
				if (!f->enabled())
					continue;

				f->first_derivative(x, tmp);

				grad += tmp;
			}
		}
	}

	void NLProblem::hessian(const TVector &x, THessian &hessian)
	{
		FullNLProblem::hessian(reduced_to_full(x), hessian);

		if (full_size() != current_size())
		{
			hessian = Q2_.transpose() * hessian * Q2_;
		}
		else
		{
			THessian tmp;
			for (const auto &f : penalty_forms_)
			{
				if (!f->enabled())
					continue;
				f->second_derivative(x, tmp);
				hessian += tmp;
			}
		}
	}

	void NLProblem::solution_changed(const TVector &newX)
	{
		FullNLProblem::solution_changed(reduced_to_full(newX));
	}

	void NLProblem::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		FullNLProblem::post_step(polysolve::nonlinear::PostStepData(data.iter_num, data.solver_info, reduced_to_full(data.x), reduced_to_full(data.grad)));

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
		return reduced;
	}

	NLProblem::TVector NLProblem::reduced_to_full(const TVector &reduced) const
	{
		// Full is already at the reduced size
		if (full_size() == current_size() || full_size() == reduced.size())
		{
			return reduced;
		}

		return Q1R1iTb_ + Q2_ * reduced;
	}

} // namespace polyfem::solver

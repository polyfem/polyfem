#include "NLProblem.hpp"

#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

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
		const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &penalty_forms)
		: FullNLProblem(forms),
		  full_size_(full_size),
		  t_(0),
		  penalty_forms_(penalty_forms)
	{
		setup_constrain_nodes();
		reduced_size_ = full_size_ - constraint_nodes_.size();

		use_reduced_size();
	}

	NLProblem::NLProblem(
		const int full_size,
		const std::shared_ptr<utils::PeriodicBoundary> &periodic_bc,
		const double t,
		const std::vector<std::shared_ptr<Form>> &forms,
		const std::vector<std::shared_ptr<AugmentedLagrangianForm>> &penalty_forms)
		: FullNLProblem(forms),
		  full_size_(full_size),
		  periodic_bc_(periodic_bc),
		  t_(t),
		  penalty_forms_(penalty_forms)
	{
		setup_constrain_nodes();
		reduced_size_ = (periodic_bc ? periodic_bc->n_periodic_dof() : full_size) - constraint_nodes_.size();

		assert(std::is_sorted(constraint_nodes_.begin(), constraint_nodes_.end()));
		assert(constraint_nodes_.size() == 0 || (constraint_nodes_.front() >= 0 && constraint_nodes_.back() < full_size_));
		use_reduced_size();
	}

	void NLProblem::setup_constrain_nodes()
	{
		constraint_nodes_.clear();

		for (const auto &f : penalty_forms_)
			constraint_nodes_.insert(constraint_nodes_.end(),
									 f->constraint_nodes().begin(),
									 f->constraint_nodes().end());
		std::sort(constraint_nodes_.begin(), constraint_nodes_.end());
		auto it = std::unique(constraint_nodes_.begin(), constraint_nodes_.end());
		constraint_nodes_.resize(std::distance(constraint_nodes_.begin(), it));
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
		return FullNLProblem::value(reduced_to_full(x));
	}

	void NLProblem::gradient(const TVector &x, TVector &grad)
	{
		TVector full_grad;
		FullNLProblem::gradient(reduced_to_full(x), full_grad);
		grad = full_to_reduced_grad(full_grad);
	}

	void NLProblem::hessian(const TVector &x, THessian &hessian)
	{
		THessian full_hessian;
		FullNLProblem::hessian(reduced_to_full(x), full_hessian);

		full_hessian_to_reduced_hessian(full_hessian, hessian);
	}

	void NLProblem::solution_changed(const TVector &newX)
	{
		FullNLProblem::solution_changed(reduced_to_full(newX));
	}

	void NLProblem::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		FullNLProblem::post_step(polysolve::nonlinear::PostStepData(data.iter_num, data.solver_info, reduced_to_full(data.x), reduced_to_full(data.grad)));

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
		TVector reduced;
		full_to_reduced_aux(constraint_nodes_, full_size(), current_size(), full, reduced);
		return reduced;
	}

	NLProblem::TVector NLProblem::full_to_reduced_grad(const TVector &full) const
	{
		TVector reduced;
		full_to_reduced_aux_grad(constraint_nodes_, full_size(), current_size(), full, reduced);
		return reduced;
	}

	NLProblem::TVector NLProblem::reduced_to_full(const TVector &reduced) const
	{
		TVector full;
		reduced_to_full_aux(constraint_nodes_, full_size(), current_size(), reduced, constraint_values(reduced), full);
		return full;
	}

	Eigen::MatrixXd NLProblem::constraint_values(const TVector &reduced) const
	{
		Eigen::MatrixXd result = Eigen::MatrixXd::Zero(full_size(), 1);

		for (const auto &form : penalty_forms_)
		{
			const auto tmp = form->target(reduced);
			if (tmp.size() > 0)
				result += tmp;
		}

		return result;
	}

	template <class FullMat, class ReducedMat>
	void NLProblem::full_to_reduced_aux(const std::vector<int> &constraint_nodes, const int full_size, const int reduced_size, const FullMat &full, ReducedMat &reduced) const
	{
		using namespace polyfem;

		// Reduced is already at the full size
		if (full_size == reduced_size || full.size() == reduced_size)
		{
			reduced = full;
			return;
		}

		assert(full.size() == full_size);
		assert(full.cols() == 1);
		reduced.resize(reduced_size, 1);

		Eigen::MatrixXd mid;
		if (periodic_bc_)
			mid = periodic_bc_->full_to_periodic(full, false);
		else
			mid = full;

		assert(std::is_sorted(constraint_nodes.begin(), constraint_nodes.end()));

		long j = 0;
		size_t k = 0;
		for (int i = 0; i < mid.size(); ++i)
		{
			if (k < constraint_nodes.size() && constraint_nodes[k] == i)
			{
				++k;
				continue;
			}

			assert(j < reduced.size());
			reduced(j++) = mid(i);
		}
	}

	template <class ReducedMat, class FullMat>
	void NLProblem::reduced_to_full_aux(const std::vector<int> &constraint_nodes, const int full_size, const int reduced_size, const ReducedMat &reduced, const Eigen::MatrixXd &rhs, FullMat &full) const
	{
		using namespace polyfem;

		// Full is already at the reduced size
		if (full_size == reduced_size || full_size == reduced.size())
		{
			full = reduced;
			return;
		}

		assert(reduced.size() == reduced_size);
		assert(reduced.cols() == 1);
		full.resize(full_size, 1);

		assert(std::is_sorted(constraint_nodes.begin(), constraint_nodes.end()));

		long j = 0;
		size_t k = 0;
		Eigen::MatrixXd mid(reduced_size + constraint_nodes.size(), 1);
		for (int i = 0; i < mid.size(); ++i)
		{
			if (k < constraint_nodes.size() && constraint_nodes[k] == i)
			{
				++k;
				mid(i) = rhs(i);
				continue;
			}

			mid(i) = reduced(j++);
		}

		full = periodic_bc_ ? periodic_bc_->periodic_to_full(full_size, mid) : mid;
	}

	template <class FullMat, class ReducedMat>
	void NLProblem::full_to_reduced_aux_grad(const std::vector<int> &constraint_nodes, const int full_size, const int reduced_size, const FullMat &full, ReducedMat &reduced) const
	{
		using namespace polyfem;

		// Reduced is already at the full size
		if (full_size == reduced_size || full.size() == reduced_size)
		{
			reduced = full;
			return;
		}

		assert(full.size() == full_size);
		assert(full.cols() == 1);
		reduced.resize(reduced_size, 1);

		Eigen::MatrixXd mid;
		if (periodic_bc_)
			mid = periodic_bc_->full_to_periodic(full, true);
		else
			mid = full;

		long j = 0;
		size_t k = 0;
		for (int i = 0; i < mid.size(); ++i)
		{
			if (k < constraint_nodes.size() && constraint_nodes[k] == i)
			{
				++k;
				continue;
			}

			reduced(j++) = mid(i);
		}
	}

	void NLProblem::full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const
	{
		// POLYFEM_SCOPED_TIMER("\tfull hessian to reduced hessian");
		THessian mid = full;

		if (periodic_bc_)
			periodic_bc_->full_to_periodic(mid);

		if (current_size() < full_size())
			utils::full_to_reduced_matrix(mid.rows(), mid.rows() - constraint_nodes_.size(), constraint_nodes_, mid, reduced);
		else
			reduced = mid;
	}
} // namespace polyfem::solver

#include "NLProblem.hpp"

#include <polyfem/io/OBJWriter.hpp>

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
		const std::vector<int> &boundary_nodes,
		const std::vector<std::shared_ptr<Form>> &forms)
		: FullNLProblem(forms),
		  boundary_nodes_(boundary_nodes),
		  full_size_(full_size),
		  reduced_size_(full_size_ - boundary_nodes.size()),
		  rhs_assembler_(nullptr),
		  local_boundary_(nullptr),
		  n_boundary_samples_(0),
		  t_(0)
	{
		use_reduced_size();
	}

	NLProblem::NLProblem(
		const int full_size,
		const std::vector<int> &boundary_nodes,
		const std::vector<mesh::LocalBoundary> &local_boundary,
		const int n_boundary_samples,
		const assembler::RhsAssembler &rhs_assembler,
		const double t,
		const std::vector<std::shared_ptr<Form>> &forms)
		: FullNLProblem(forms),
		  boundary_nodes_(boundary_nodes),
		  full_size_(full_size),
		  reduced_size_(full_size_ - boundary_nodes.size()),
		  rhs_assembler_(&rhs_assembler),
		  local_boundary_(&local_boundary),
		  n_boundary_samples_(n_boundary_samples),
		  t_(t)
	{
		assert(std::is_sorted(boundary_nodes.begin(), boundary_nodes.end()));
		assert(boundary_nodes.size() == 0 || (boundary_nodes.front() >= 0 && boundary_nodes.back() < full_size_));
		use_reduced_size();
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

	double NLProblem::max_step_size(const TVector &x0, const TVector &x1) const
	{
		return FullNLProblem::max_step_size(reduced_to_full(x0), reduced_to_full(x1));
	}

	bool NLProblem::is_step_valid(const TVector &x0, const TVector &x1) const
	{
		return FullNLProblem::is_step_valid(reduced_to_full(x0), reduced_to_full(x1));
	}

	bool NLProblem::is_step_collision_free(const TVector &x0, const TVector &x1) const
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
		grad = full_to_reduced(full_grad);
	}

	void NLProblem::hessian(const TVector &x, THessian &hessian)
	{
		THessian full_hessian;
		FullNLProblem::hessian(reduced_to_full(x), full_hessian);
		assert(full_hessian.rows() == full_size());
		assert(full_hessian.cols() == full_size());
		utils::full_to_reduced_matrix(full_size(), current_size(), boundary_nodes_, full_hessian, hessian);
	}

	void NLProblem::solution_changed(const TVector &newX)
	{
		FullNLProblem::solution_changed(reduced_to_full(newX));
	}

	void NLProblem::post_step(const int iter_num, const TVector &x)
	{
		FullNLProblem::post_step(iter_num, reduced_to_full(x));

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

	void NLProblem::set_apply_DBC(const TVector &x, const bool val)
	{
		TVector full = reduced_to_full(x);
		for (auto &form : forms_)
			form->set_apply_DBC(full, val);
	}

	NLProblem::TVector NLProblem::full_to_reduced(const TVector &full) const
	{
		TVector reduced;
		full_to_reduced_aux(boundary_nodes_, full_size(), current_size(), full, reduced);
		return reduced;
	}

	NLProblem::TVector NLProblem::reduced_to_full(const TVector &reduced) const
	{
		TVector full;
		reduced_to_full_aux(boundary_nodes_, full_size(), current_size(), reduced, boundary_values(), full);
		return full;
	}

	Eigen::MatrixXd NLProblem::boundary_values() const
	{
		Eigen::MatrixXd result = Eigen::MatrixXd::Zero(full_size(), 1);
		// rhs_assembler->set_bc(*local_boundary_, boundary_nodes_, n_boundary_samples_, local_neumann_boundary_, result, t_);
		rhs_assembler_->set_bc(*local_boundary_, boundary_nodes_, n_boundary_samples_, std::vector<mesh::LocalBoundary>(), result, Eigen::MatrixXd(), t_);
		return result;
	}

	template <class FullMat, class ReducedMat>
	void NLProblem::full_to_reduced_aux(const std::vector<int> &boundary_nodes, const int full_size, const int reduced_size, const FullMat &full, ReducedMat &reduced)
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

		assert(std::is_sorted(boundary_nodes.begin(), boundary_nodes.end()));

		long j = 0;
		size_t k = 0;
		for (int i = 0; i < full.size(); ++i)
		{
			if (k < boundary_nodes.size() && boundary_nodes[k] == i)
			{
				++k;
				continue;
			}

			reduced(j++) = full(i);
		}
		assert(j == reduced.size());
	}

	template <class ReducedMat, class FullMat>
	void NLProblem::reduced_to_full_aux(const std::vector<int> &boundary_nodes, const int full_size, const int reduced_size, const ReducedMat &reduced, const Eigen::MatrixXd &rhs, FullMat &full)
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

		assert(std::is_sorted(boundary_nodes.begin(), boundary_nodes.end()));

		long j = 0;
		size_t k = 0;
		for (int i = 0; i < full.size(); ++i)
		{
			if (k < boundary_nodes.size() && boundary_nodes[k] == i)
			{
				++k;
				full(i) = rhs(i);
				continue;
			}

			full(i) = reduced(j++);
		}

		assert(j == reduced.size());
	}
} // namespace polyfem::solver

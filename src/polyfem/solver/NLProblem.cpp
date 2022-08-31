#include "NLProblem.hpp"

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>

#include <igl/write_triangle_mesh.h>

static bool disable_collision = false;

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

namespace polyfem
{
	using namespace assembler;
	using namespace io;
	using namespace utils;
	namespace solver
	{
		using namespace polysolve;

		NLProblem::NLProblem(const State &state, const assembler::RhsAssembler &rhs_assembler, std::vector<std::shared_ptr<Form>> &forms)
			: state_(state), rhs_assembler_(rhs_assembler),
			  full_size((state.assembler.is_mixed(state.formulation()) ? state.n_pressure_bases : 0) + state.n_bases * state.mesh->dimension()),
			  actual_reduced_size(full_size - state.boundary_nodes.size()),
			  forms_(forms)
		{
			t_ = 0;
			assert(!state.assembler.is_mixed(state.formulation()));
			set_full_size(false);
		}

		void NLProblem::init(const TVector &full)
		{
			for (auto &f : forms_)
				f->init(full);
		}

		void NLProblem::set_project_to_psd(bool val)
		{
			for (auto &f : forms_)
				f->set_project_to_psd(val);
		}

		void NLProblem::init_lagging(const TVector &x)
		{
			TVector full;
			reduced_to_full(x, full);
			for (auto &f : forms_)
				f->init_lagging(full);
		}

		void NLProblem::update_lagging(const TVector &x)
		{
			TVector full;
			reduced_to_full(x, full);
			for (auto &f : forms_)
				f->update_lagging(full);
		}

		void NLProblem::update_quantities(const double t, const TVector &x)
		{
			t_ = t;
			TVector full;
			reduced_to_full(x, full);

			for (auto &f : forms_)
				f->update_quantities(t, full);
		}

		void NLProblem::line_search_begin(const TVector &x0, const TVector &x1)
		{
			Eigen::MatrixXd full0, full1;
			reduced_to_full(x0, full0);
			reduced_to_full(x1, full1);

			for (auto &f : forms_)
				f->line_search_begin(full0, full1);
		}

		void NLProblem::line_search_end()
		{
			for (auto &f : forms_)
				f->line_search_end();
		}

		double NLProblem::max_step_size(const TVector &x0, const TVector &x1)
		{
			Eigen::MatrixXd full0, full1;
			reduced_to_full(x0, full0);
			reduced_to_full(x1, full1);

			double step = 1;
			for (auto &f : forms_)
				step = std::min(step, f->max_step_size(full0, full1));

			return step;
		}

		bool NLProblem::is_step_valid(const TVector &x0, const TVector &x1)
		{
			TVector full0, full1;
			reduced_to_full(x0, full0);
			reduced_to_full(x1, full1);
			for (auto &f : forms_)
				if (!f->is_step_valid(full0, full1))
					return false;

			return true;
		}

		double NLProblem::value(const TVector &x)
		{
			// TODO: removed fearure const bool only_elastic

			TVector full;
			reduced_to_full(x, full);

			double fvalue = 0;
			for (int i = 0; i < forms_.size(); ++i)
			{
				const auto &f = forms_[i];
				if (!f->enabled())
					continue;
				fvalue += f->value(full);
			}
			return fvalue;
		}

		void NLProblem::gradient(const TVector &x, TVector &gradv)
		{
			TVector full, tmp;
			reduced_to_full(x, full);
			TVector fgrad(full.size());
			fgrad.setZero();
			for (int i = 0; i < forms_.size(); ++i)
			{
				const auto &f = forms_[i];
				if (!f->enabled())
					continue;
				f->first_derivative(full, tmp);
				fgrad += tmp;
			}

			full_to_reduced(fgrad, gradv);
		}

		void NLProblem::hessian(const TVector &x, THessian &hessian)
		{
			THessian full_hessian;
			hessian_full(x, full_hessian);
			full_to_reduced_matrix(full_size, reduced_size, state_.boundary_nodes, full_hessian, hessian);
		}

		void NLProblem::hessian_full(const TVector &x, THessian &hessian)
		{
			TVector full;
			reduced_to_full(x, full);

			THessian tmp(full_size, full_size);
			hessian.resize(full_size, full_size);
			hessian.setZero();
			for (int i = 0; i < forms_.size(); ++i)
			{
				const auto &f = forms_[i];
				if (!f->enabled())
					continue;
				f->second_derivative(full, tmp);
				hessian += tmp;
			}
			assert(hessian.rows() == full_size);
			assert(hessian.cols() == full_size);
		}

		void NLProblem::solution_changed(const TVector &newX)
		{
			Eigen::MatrixXd newFull;
			reduced_to_full(newX, newFull);

			for (auto &f : forms_)
				f->solution_changed(newFull);
		}

		void NLProblem::post_step(const int iter_num, const TVector &x)
		{
			TVector full;
			reduced_to_full(x, full);

			for (auto &f : forms_)
				f->post_step(iter_num, full);
		}
	} // namespace solver
} // namespace polyfem

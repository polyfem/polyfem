#include "NLHomogenizationProblem.hpp"

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/OBJ_IO.hpp>

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
	using namespace utils;

	namespace solver
	{
		using namespace polysolve;

		NLHomogenizationProblem::NLHomogenizationProblem(State &state, const RhsAssembler &rhs_assembler, const bool no_reduced)
			: state(state), assembler(state.assembler), rhs_assembler(rhs_assembler),
			  full_size(state.n_bases * state.mesh->dimension()),
			  reduced_size(full_size - (no_reduced ? 0 : state.boundary_nodes.size())),
			  project_to_psd(false)
		{
			assert(!assembler.is_mixed(state.formulation()));
			index[0] = 0;
			index[1] = 0;
		}

		const Eigen::MatrixXd &NLHomogenizationProblem::current_rhs()
		{
			if (!rhs_computed)
			{
				// rhs_assembler.compute_energy_grad(state.local_boundary, state.boundary_nodes, state.density, state.n_boundary_samples(), state.local_neumann_boundary, state.rhs, t, _current_rhs);
				_current_rhs.setZero(full_size, 1);
				rhs_computed = true;

				assert(_current_rhs.size() == full_size);
				// rhs_assembler.set_bc(std::vector<mesh::LocalBoundary>(), std::vector<int>(), state.n_boundary_samples(), state.local_neumann_boundary, _current_rhs, t);

				if (reduced_size != full_size)
				{
					logger().error("Homogenization doesn't support Dirichlet BC!");
					throw std::runtime_error("Homogenization doesn't support Dirichlet BC!");
					// rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.n_boundary_samples(), state.local_neumann_boundary, _current_rhs, t);
					// rhs_assembler.set_bc(state.local_boundary, state.boundary_nodes, state.n_boundary_samples(), std::vector<mesh::LocalBoundary>(), _current_rhs, t);
				}
			}

			return _current_rhs;
		}

		bool NLHomogenizationProblem::is_step_valid(const TVector &x0, const TVector &x1)
		{
			TVector grad = TVector::Zero(reduced_size);
			gradient(x1, grad, true);

			if (std::isnan(grad.norm()))
				return false;

			return true;
		}

		double NLHomogenizationProblem::value(const TVector &x)
		{
			return value(x, false);
		}

		double NLHomogenizationProblem::value(const TVector &x, const bool only_elastic)
		{
			TVector full;
			reduced_to_full(x, full);

			const double elastic_energy = state.assemble_neohookean_homogenization_energy(full, index[0], index[1]);

			return elastic_energy;
		}

		void NLHomogenizationProblem::gradient(const TVector &x, TVector &gradv)
		{
			gradient(x, gradv, false);
		}

		void NLHomogenizationProblem::gradient(const TVector &x, TVector &gradv, const bool only_elastic)
		{
			Eigen::MatrixXd grad;
			gradient_no_rhs(x, grad, only_elastic);

			full_to_reduced(grad, gradv);
		}

		void NLHomogenizationProblem::gradient_no_rhs(const TVector &x, Eigen::MatrixXd &grad, const bool only_elastic)
		{
			TVector full;
			reduced_to_full(x, full);

			// const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
			// assembler.assemble_energy_gradient(rhs_assembler.formulation(), state.mesh->is_volume(), state.n_bases, state.bases, gbases, state.ass_vals_cache, full, grad);
			state.assemble_neohookean_homogenization_gradient(grad, full, index[0], index[1]);

			assert(grad.size() == full_size);
		}

		void NLHomogenizationProblem::hessian(const TVector &x, THessian &hessian)
		{
			THessian full_hessian;
			hessian_full(x, full_hessian);
			full_hessian_to_reduced_hessian(full_hessian, hessian);

			// lagrange multiplier for periodic bc
			if (reduced_size == full_size && !state.problem->is_time_dependent())
			{
				if (state.args["boundary_conditions"]["periodic_boundary"].get<bool>())
					state.remove_pure_periodic_singularity(hessian);
				else
					state.remove_pure_neumann_singularity(hessian);
			}
		}

		void NLHomogenizationProblem::hessian_full(const TVector &x, THessian &hessian)
		{
			// elastic_energy + body_energy;

			TVector full;
			reduced_to_full(x, full);

			state.assemble_neohookean_homogenization_hessian(hessian, mat_cache, full, index[0], index[1]);

			assert(hessian.rows() == full_size);
			assert(hessian.cols() == full_size);
		}

		void NLHomogenizationProblem::full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const
		{
			POLYFEM_SCOPED_TIMER("\tfull hessian to reduced hessian");

			if (reduced_size == full_size || reduced_size == full.rows())
			{
				assert(reduced_size == full.rows() && reduced_size == full.cols());
				reduced = full;
				return;
			}

			Eigen::VectorXi indices(full_size);
			int index = 0;
			size_t kk = 0;
			for (int i = 0; i < full_size; ++i)
			{
				if (kk < state.boundary_nodes.size() && state.boundary_nodes[kk] == i)
				{
					++kk;
					indices(i) = -1;
				}
				else
				{
					indices(i) = index++;
				}
			}
			assert(index == reduced_size);

			std::vector<Eigen::Triplet<double>> entries;
			entries.reserve(full.nonZeros()); // Conservative estimate
			for (int k = 0; k < full.outerSize(); ++k)
			{
				if (indices(k) < 0)
					continue;

				for (THessian::InnerIterator it(full, k); it; ++it)
				{
					assert(it.col() == k);
					if (indices(it.row()) < 0 || indices(it.col()) < 0)
						continue;

					assert(indices(it.row()) >= 0);
					assert(indices(it.col()) >= 0);

					entries.emplace_back(indices(it.row()), indices(it.col()), it.value());
				}
			}

			reduced.resize(reduced_size, reduced_size);
			reduced.setFromTriplets(entries.begin(), entries.end());
			reduced.makeCompressed();
		}
	} // namespace solver
} // namespace polyfem

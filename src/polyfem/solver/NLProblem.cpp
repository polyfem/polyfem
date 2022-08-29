#include "NLProblem.hpp"

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/OBJ_IO.hpp>

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

// map BroadPhaseMethod values to JSON as strings
namespace ipc
{
	NLOHMANN_JSON_SERIALIZE_ENUM(
		ipc::BroadPhaseMethod,
		{{ipc::BroadPhaseMethod::HASH_GRID, "hash_grid"}, // also default
		 {ipc::BroadPhaseMethod::HASH_GRID, "HG"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "brute_force"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "BF"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "spatial_hash"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "SH"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "sweep_and_tiniest_queue"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "STQ"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "sweep_and_tiniest_queue_gpu"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "STQ_GPU"}});
} // namespace ipc

namespace polyfem
{
	using namespace assembler;
	using namespace utils;
	® namespace solver
	{
		using namespace polysolve;

		NLProblem::NLProblem(const State &state, std::vector<std::shared_ptr<Form>> &forms, const bool no_reduced)
			: state_(state),
			  full_size((state.assembler.is_mixed(state.formulation()) ? state.n_pressure_bases : 0) + state.n_bases * state.mesh->dimension()),
			  reduced_size(full_size - (no_reduced ? 0 : state.boundary_nodes.size())),
			  forms_(form)
		{
			assert(!assembler.is_mixed(state.formulation()));
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

		double NLProblem::compute_lagging_error(const TVector &x)
		{
			// TODO: REFACTOR ME
			//  Check || ∇B(xᵗ⁺¹) - h² Σ F(xᵗ⁺¹, λᵗ⁺¹, Tᵗ⁺¹)|| ≦ ϵ_d
			//      ≡ || ∇B(xᵗ⁺¹) + ∇D(xᵗ⁺¹, λᵗ⁺¹, Tᵗ⁺¹)|| ≤ ϵ_d
			TVector grad;
			gradient(x, grad);
			return grad.norm();
		}

		bool NLProblem::lagging_converged(const TVector &x)
		{
			// TODO: REFACTOR ME
			double tol = state.args["solver"]["contact"].value("friction_convergence_tol", 1e-2);
			double grad_norm = compute_lagging_error(x);
			logger().debug("Lagging convergece grad_norm={:g} tol={:g}", grad_norm, tol);
			return grad_norm <= tol;
		}

		void NLProblem::update_quantities(const double t, const TVector &x)
		{
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

		bool NLProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
		{
			// TODO: DELETE ME BELOW
			if (disable_collision || !state.is_contact_enabled())
				return true;

			// if (!state.problem->is_time_dependent())
			// return false;

			Eigen::MatrixXd displaced0, displaced1;
			reduced_to_full_displaced_points(x0, displaced0);
			reduced_to_full_displaced_points(x1, displaced1);

			// Skip CCD if the displacement is zero.
			if ((displaced1 - displaced0).lpNorm<Eigen::Infinity>() == 0.0)
			{
				// Assumes initially intersection-free
				assert(is_intersection_free(x0));
				return true;
			}

			// OBJWriter::save("0.obj", state.collision_mesh.vertices(displaced0), state.collision_mesh.edges(), state.collision_mesh.faces());
			// OBJWriter::save("1.obj", state.collision_mesh.vertices(displaced1), state.collision_mesh.edges(), state.collision_mesh.faces());

			bool is_valid;
			if (_use_cached_candidates)
				is_valid = ipc::is_step_collision_free(
					_candidates, state.collision_mesh,
					state.collision_mesh.vertices(displaced0),
					state.collision_mesh.vertices(displaced1),
					_ccd_tolerance, _ccd_max_iterations);
			else
				is_valid = ipc::is_step_collision_free(
					state.collision_mesh,
					state.collision_mesh.vertices(displaced0),
					state.collision_mesh.vertices(displaced1),
					_broad_phase_method, _ccd_tolerance, _ccd_max_iterations);

			return is_valid;
		}

		bool NLProblem::is_intersection_free(const TVector &x)
		{
			// TODO: DELETE ME BELOW
			if (disable_collision || !state.is_contact_enabled())
				return true;

			Eigen::MatrixXd displaced;
			reduced_to_full_displaced_points(x, displaced);

			return !ipc::has_intersections(state.collision_mesh, state.collision_mesh.vertices(displaced));
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
			return value(x, false);
		}

		double NLProblem::value(const TVector &x, const bool only_elastic)
		{
			TVector full;
			reduced_to_full(x, full);

			double fvalue = 0;
			for (int i = 0; i < forms_.size(); ++i)
			{
				const auto &f = forms_[i];
				if (only_elastic && i == 2)
					continue;
				fvalue += f->value(full);
			}
			return fvalue;
		}

		void NLProblem::gradient(const TVector &x, TVector &gradv)
		{
			gradient(x, gradv, false);
		}

		void NLProblem::gradient(const TVector &x, TVector &gradv, const bool only_elastic)
		{
			TVector full, tmp;
			reduced_to_full(x, full);
			TVector fgrad(full.size());
			for (int i = 0; i < forms_.size(); ++i)
			{
				const auto &f = forms_[i];
				if (only_elastic && i == 2)
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
			full_hessian_to_reduced_hessian(full_hessian, hessian);
		}

		void NLProblem::hessian_full(const TVector &x, THessian &hessian)
		{
			// scaling * (elastic_energy + body_energy) + intertia_energy + _barrier_stiffness * collision_energy;

			TVector full;
			reduced_to_full(x, full);

			THessian tmp(full_size, full_size);
			hessian.resize(full_size, full_size);
			for (int i = 0; i < forms_.size(); ++i)
			{
				const auto &f = forms_[i];
				f->second_derivative(full, tmp);
				hessian += tmp;
			}
			assert(hessian.rows() == full_size);
			assert(hessian.cols() == full_size);
		}

		void NLProblem::full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const
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

		void NLProblem::save_raw(const std::string &x_path, const std::string &v_path, const std::string &a_path) const
		{
			time_integrator()->save_raw(x_path, v_path, a_path);
		}
	} // namespace solver
} // namespace polyfem

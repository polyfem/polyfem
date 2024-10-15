#include "NormalAdhesionForm.hpp"

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/io/OBJWriter.hpp>

#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>

#include <igl/writePLY.h>

namespace polyfem::solver
{
	NormalAdhesionForm::NormalAdhesionForm(const ipc::CollisionMesh &collision_mesh,
							 const double dhat,
							 const double avg_mass,
							 const bool use_convergent_formulation,
							 const bool use_adaptive_barrier_stiffness,
							 const bool is_time_dependent,
							 const bool enable_shape_derivatives,
							 const ipc::BroadPhaseMethod broad_phase_method,
							 const double ccd_tolerance,
							 const int ccd_max_iterations)
		: collision_mesh_(collision_mesh),
		  dhat_(dhat),
		  use_adaptive_barrier_stiffness_(use_adaptive_barrier_stiffness),
		  avg_mass_(avg_mass),
		  is_time_dependent_(is_time_dependent),
		  enable_shape_derivatives_(enable_shape_derivatives),
		  broad_phase_method_(broad_phase_method),
		  ccd_tolerance_(ccd_tolerance),
		  ccd_max_iterations_(ccd_max_iterations),
		  barrier_potential_(dhat)
	{
		assert(dhat_ > 0);
		assert(ccd_tolerance > 0);

		prev_distance_ = -1;
		//collision_set_.set_use_convergent_formulation(use_convergent_formulation);
		collision_set_.set_use_improved_max_approximator(use_convergent_formulation);
		collision_set_.set_use_area_weighting(use_convergent_formulation);
		collision_set_.set_enable_shape_derivatives(enable_shape_derivatives);
	}

	void NormalAdhesionForm::init(const Eigen::VectorXd &x)
	{
		update_collision_set(compute_displaced_surface(x));
	}

	void NormalAdhesionForm::force_shape_derivative(const ipc::NormalCollisions &collision_set, const Eigen::MatrixXd &solution, const Eigen::VectorXd &adjoint_sol, Eigen::VectorXd &term)
	{
		// Eigen::MatrixXd U = collision_mesh_.vertices(utils::unflatten(solution, collision_mesh_.dim()));
		// Eigen::MatrixXd X = collision_mesh_.vertices(boundary_nodes_pos_);
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(solution);

		StiffnessMatrix dq_h = collision_mesh_.to_full_dof(barrier_potential_.shape_derivative(collision_set, collision_mesh_, displaced_surface));
		term = barrier_stiffness() * dq_h.transpose() * adjoint_sol;
	}

	void NormalAdhesionForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		update_collision_set(compute_displaced_surface(x));
	}

	Eigen::MatrixXd NormalAdhesionForm::compute_displaced_surface(const Eigen::VectorXd &x) const
	{
		return collision_mesh_.displace_vertices(utils::unflatten(x, collision_mesh_.dim()));
	}

	void NormalAdhesionForm::update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy)
	{
		if (!use_adaptive_barrier_stiffness())
			return;

		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);

		// The adative stiffness is designed for the non-convergent formulation,
		// so we need to compute the gradient of the non-convergent barrier.
		// After we can map it to a good value for the convergent formulation.
		ipc::NormalCollisions nonconvergent_constraints;
		//nonconvergent_constraints.set_use_convergent_formulation(false);
		nonconvergent_constraints.set_use_area_weighting(false);
		nonconvergent_constraints.set_use_improved_max_approximator(false);
		nonconvergent_constraints.build(
			collision_mesh_, displaced_surface, dhat_, dmin_, broad_phase_method_);
		Eigen::VectorXd grad_barrier = barrier_potential_.gradient(
			nonconvergent_constraints, collision_mesh_, displaced_surface);
		grad_barrier = collision_mesh_.to_full_dof(grad_barrier);

		barrier_stiffness_ = ipc::initial_barrier_stiffness(
			ipc::world_bbox_diagonal_length(displaced_surface), barrier_potential_.barrier(), dhat_, avg_mass_,
			grad_energy, grad_barrier, max_barrier_stiffness_);

		if (use_convergent_formulation())
		{
			double scaling_factor = 0;
			if (!nonconvergent_constraints.empty())
			{
				const double nonconvergent_potential = barrier_potential_(
					nonconvergent_constraints, collision_mesh_, displaced_surface);

				update_collision_set(displaced_surface);
				const double convergent_potential = barrier_potential_(
					collision_set_, collision_mesh_, displaced_surface);

				scaling_factor = nonconvergent_potential / convergent_potential;
			}
			else
			{
				// Hardcoded difference between the non-convergent and convergent barrier
				scaling_factor = dhat_ * std::pow(dhat_ + 2 * dmin_, 2);
			}
			barrier_stiffness_ *= scaling_factor;
			max_barrier_stiffness_ *= scaling_factor;
		}

		// The barrier stiffness is choosen based on including the acceleration scaling,
		// but the acceleration scaling will be applied later. Therefore, we need to remove it.
		barrier_stiffness_ /= weight_;
		max_barrier_stiffness_ /= weight_;

		logger().debug(
			"Setting adaptive barrier stiffness to {} (max barrier stiffness: {})",
			barrier_stiffness(), max_barrier_stiffness_);
	}

	void NormalAdhesionForm::update_collision_set(const Eigen::MatrixXd &displaced_surface)
	{
		// Store the previous value used to compute the constraint set to avoid duplicate computation.
		static Eigen::MatrixXd cached_displaced_surface;
		if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
			return;

		if (use_cached_candidates_)
			collision_set_.build(
				candidates_, collision_mesh_, displaced_surface, dhat_);
		else
			collision_set_.build(
				collision_mesh_, displaced_surface, dhat_, dmin_, broad_phase_method_);
		cached_displaced_surface = displaced_surface;
	}

	double NormalAdhesionForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return barrier_potential_(collision_set_, collision_mesh_, compute_displaced_surface(x));
	}

	Eigen::VectorXd NormalAdhesionForm::value_per_element_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd V = compute_displaced_surface(x);
		assert(V.rows() == collision_mesh_.num_vertices());

		const size_t num_vertices = collision_mesh_.num_vertices();

		if (collision_set_.empty())
		{
			return Eigen::VectorXd::Zero(collision_mesh_.full_num_vertices());
		}

		const Eigen::MatrixXi &E = collision_mesh_.edges();
		const Eigen::MatrixXi &F = collision_mesh_.faces();

		auto storage = utils::create_thread_storage<Eigen::VectorXd>(Eigen::VectorXd::Zero(num_vertices));

		utils::maybe_parallel_for(collision_set_.size(), [&](int start, int end, int thread_id) {
			Eigen::VectorXd &local_storage = utils::get_local_thread_storage(storage, thread_id);

			for (size_t i = start; i < end; i++)
			{
				// Quadrature weight is premultiplied by compute_potential
				const double potential = barrier_potential_(collision_set_[i], collision_set_[i].dof(V, E, F));

				const int n_v = collision_set_[i].num_vertices();
				const std::array<long, 4> vis = collision_set_[i].vertex_ids(E, F);
				for (int j = 0; j < n_v; j++)
				{
					assert(0 <= vis[j] && vis[j] < num_vertices);
					local_storage[vis[j]] += potential / n_v;
				}
			}
		});

		Eigen::VectorXd out = Eigen::VectorXd::Zero(num_vertices);
		for (const auto &local_potential : storage)
		{
			out += local_potential;
		}

		Eigen::VectorXd out_full = Eigen::VectorXd::Zero(collision_mesh_.full_num_vertices());
		for (int i = 0; i < out.size(); i++)
			out_full[collision_mesh_.to_full_vertex_id(i)] = out[i];

		assert(std::abs(value_unweighted(x) - out_full.sum()) < std::max(1e-10 * out_full.sum(), 1e-10));

		return out_full;
	}

	void NormalAdhesionForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = barrier_potential_.gradient(collision_set_, collision_mesh_, compute_displaced_surface(x));
		gradv = collision_mesh_.to_full_dof(gradv);
	}

	void NormalAdhesionForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("barrier hessian");

		ipc::PSDProjectionMethod psd_projection_method;

		if (project_to_psd_) {
			psd_projection_method = ipc::PSDProjectionMethod::CLAMP;
		} else {
			psd_projection_method = ipc::PSDProjectionMethod::NONE;
		}

		hessian = barrier_potential_.hessian(collision_set_, collision_mesh_, compute_displaced_surface(x), psd_projection_method);
		hessian = collision_mesh_.to_full_dof(hessian);
	}

	void NormalAdhesionForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		update_collision_set(compute_displaced_surface(new_x));
	}

	double NormalAdhesionForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// Extract surface only
		const Eigen::MatrixXd V0 = compute_displaced_surface(x0);
		const Eigen::MatrixXd V1 = compute_displaced_surface(x1);

		if (save_ccd_debug_meshes)
		{
			const Eigen::MatrixXi E = collision_mesh_.dim() == 2 ? Eigen::MatrixXi() : collision_mesh_.edges();
			const Eigen::MatrixXi &F = collision_mesh_.faces();
			igl::writePLY(resolve_output_path("debug_ccd_0.ply"), V0, F, E);
			igl::writePLY(resolve_output_path("debug_ccd_1.ply"), V1, F, E);
		}

		double max_step;
		const ipc::TightInclusionCCD tight_inclusion_ccd(ccd_tolerance_, ccd_max_iterations_);
		if (use_cached_candidates_ && broad_phase_method_ != ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE)
			max_step = candidates_.compute_collision_free_stepsize(
				collision_mesh_, V0, V1, dmin_, tight_inclusion_ccd);
		else
			max_step = ipc::compute_collision_free_stepsize(
				collision_mesh_, V0, V1, dmin_, broad_phase_method_, tight_inclusion_ccd);

		if (save_ccd_debug_meshes && ipc::has_intersections(collision_mesh_, (V1 - V0) * max_step + V0, broad_phase_method_))
		{
			log_and_throw_error("Taking max_step results in intersections (max_step={})", max_step);
		}

#ifndef NDEBUG
		// This will check for static intersections as a failsafe. Not needed if we use our conservative CCD.
		Eigen::MatrixXd V_toi = (V1 - V0) * max_step + V0;

		while (ipc::has_intersections(collision_mesh_, V_toi, broad_phase_method_))
		{
			logger().error("Taking max_step results in intersections (max_step={:g})", max_step);
			max_step /= 2.0;

			const double Linf = (V_toi - V0).lpNorm<Eigen::Infinity>();
			if (max_step <= 0 || Linf == 0)
				log_and_throw_error("Unable to find an intersection free step size (max_step={:g} L∞={:g})", max_step, Linf);

			V_toi = (V1 - V0) * max_step + V0;
		}
#endif

		return max_step;
	}

	void NormalAdhesionForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		candidates_.build(
			collision_mesh_,
			compute_displaced_surface(x0),
			compute_displaced_surface(x1),
			/*inflation_radius=*/dhat_ / 2,
			broad_phase_method_);

		use_cached_candidates_ = true;
	}

	void NormalAdhesionForm::line_search_end()
	{
		candidates_.clear();
		use_cached_candidates_ = false;
	}

	void NormalAdhesionForm::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		if (data.iter_num == 0)
			return;

		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(data.x);

		const double curr_distance = collision_set_.compute_minimum_distance(collision_mesh_, displaced_surface);

		if (use_adaptive_barrier_stiffness_)
		{
			if (is_time_dependent_)
			{
				const double prev_barrier_stiffness = barrier_stiffness();

				barrier_stiffness_ = ipc::update_barrier_stiffness(
					prev_distance_, curr_distance, max_barrier_stiffness_,
					barrier_stiffness(), ipc::world_bbox_diagonal_length(displaced_surface));

				if (barrier_stiffness() != prev_barrier_stiffness)
				{
					polyfem::logger().debug(
						"updated barrier stiffness from {:g} to {:g} (max barrier stiffness: )",
						prev_barrier_stiffness, barrier_stiffness(), max_barrier_stiffness_);
				}
			}
			else
			{
				// TODO: missing feature
				// update_barrier_stiffness(data.x);
			}
		}

		prev_distance_ = curr_distance;
	}

	bool NormalAdhesionForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		const auto displaced0 = compute_displaced_surface(x0);
		const auto displaced1 = compute_displaced_surface(x1);

		// Skip CCD if the displacement is zero.
		if ((displaced1 - displaced0).lpNorm<Eigen::Infinity>() == 0.0)
		{
			// Assumes initially intersection-free
			return true;
		}

		bool is_valid;
		const ipc::TightInclusionCCD tight_inclusion_ccd(ccd_tolerance_, ccd_max_iterations_);
		if (use_cached_candidates_)
			is_valid = candidates_.is_step_collision_free(
				collision_mesh_, displaced0, displaced1, dmin_,
				tight_inclusion_ccd);
		else
			is_valid = ipc::is_step_collision_free(
				collision_mesh_, displaced0, displaced1, dmin_, broad_phase_method_,
				tight_inclusion_ccd);

		return is_valid;
	}
} // namespace polyfem::solver

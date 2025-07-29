#include "ContactForm.hpp"

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
	ContactForm::ContactForm(const ipc::CollisionMesh &collision_mesh,
							 const double dhat,
							 const double avg_mass,
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
		  broad_phase_(ipc::build_broad_phase(broad_phase_method)),
		  tight_inclusion_ccd_(ccd_tolerance, ccd_max_iterations)
	{
		assert(dhat_ > 0);
		assert(ccd_tolerance > 0);

		prev_distance_ = -1;
	}

	void ContactForm::init(const Eigen::VectorXd &x)
	{
		update_collision_set(compute_displaced_surface(x));
	}

	void ContactForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		update_collision_set(compute_displaced_surface(x));
	}

	Eigen::MatrixXd ContactForm::compute_displaced_surface(const Eigen::VectorXd &x) const
	{
		return collision_mesh_.displace_vertices(utils::unflatten(x, collision_mesh_.dim()));
	}

	void ContactForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		update_collision_set(compute_displaced_surface(new_x));
	}

	double ContactForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
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
		if (use_cached_candidates_ && broad_phase_method_ != ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE)
			max_step = candidates_.compute_collision_free_stepsize(
				collision_mesh_, V0, V1, dmin_, tight_inclusion_ccd_);
		else
			max_step = ipc::compute_collision_free_stepsize(
				collision_mesh_, V0, V1, dmin_, broad_phase_, tight_inclusion_ccd_);

		if (save_ccd_debug_meshes && ipc::has_intersections(collision_mesh_, (V1 - V0) * max_step + V0, broad_phase_))
		{
			log_and_throw_error("Taking max_step results in intersections (max_step={})", max_step);
		}

#ifndef NDEBUG
		// This will check for static intersections as a failsafe. Not needed if we use our conservative CCD.
		Eigen::MatrixXd V_toi = (V1 - V0) * max_step + V0;

		while (ipc::has_intersections(collision_mesh_, V_toi, broad_phase_))
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

	void ContactForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		candidates_.build(
			collision_mesh_,
			compute_displaced_surface(x0),
			compute_displaced_surface(x1),
			/*inflation_radius=*/barrier_support_size() / 2,
			broad_phase_);

		use_cached_candidates_ = true;
	}

	void ContactForm::line_search_end()
	{
		candidates_.clear();
		use_cached_candidates_ = false;
	}

	bool ContactForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
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
		if (use_cached_candidates_)
			is_valid = candidates_.is_step_collision_free(
				collision_mesh_, displaced0, displaced1, dmin_,
				tight_inclusion_ccd_);
		else
			is_valid = ipc::is_step_collision_free(
				collision_mesh_, displaced0, displaced1, dmin_, broad_phase_,
				tight_inclusion_ccd_);

		return is_valid;
	}
} // namespace polyfem::solver

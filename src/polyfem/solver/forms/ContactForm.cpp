#include "ContactForm.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/Logger.hpp>

#include <polyfem/io/OBJWriter.hpp>

#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>

namespace polyfem::solver
{
	ContactForm::ContactForm(const State &state,
							 const double dhat,
							 const bool use_adaptive_barrier_stiffness,
							 const bool is_time_dependent,
							 const ipc::BroadPhaseMethod broad_phase_method,
							 const double ccd_tolerance,
							 const int ccd_max_iterations)
		: state_(state),
		  dhat_(dhat),
		  use_adaptive_barrier_stiffness_(use_adaptive_barrier_stiffness),
		  is_time_dependent_(is_time_dependent),
		  broad_phase_method_(broad_phase_method),
		  ccd_tolerance_(ccd_tolerance),
		  ccd_max_iterations_(ccd_max_iterations)
	{
		assert(dhat_ > 0);
		assert(ccd_tolerance > 0);

		prev_distance_ = -1;
	}

	void ContactForm::init(const Eigen::VectorXd &x)
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);
		update_constraint_set(displaced_surface);
	}

	void ContactForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);
		update_constraint_set(displaced_surface);
	}

	Eigen::MatrixXd ContactForm::compute_displaced_surface(const Eigen::VectorXd &x) const
	{
		return state_.collision_mesh.displace_vertices(utils::unflatten(x, state_.mesh->dimension()));
	}

	void ContactForm::update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy)
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);

		Eigen::VectorXd grad_barrier = ipc::compute_barrier_potential_gradient(
			state_.collision_mesh, displaced_surface, constraint_set_, dhat_);
		grad_barrier = state_.collision_mesh.to_full_dof(grad_barrier);

		weight_ = ipc::initial_barrier_stiffness(
			ipc::world_bbox_diagonal_length(displaced_surface), dhat_, state_.avg_mass,
			grad_energy, grad_barrier, max_barrier_stiffness_);

		logger().debug("adaptive barrier form stiffness {}", weight_);
	}

	void ContactForm::update_constraint_set(const Eigen::MatrixXd &displaced_surface)
	{
		// Store the previous value used to compute the constraint set to avoid duplicate computation.
		static Eigen::MatrixXd cached_displaced_surface;
		if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
			return;

		if (use_cached_candidates_)
			constraint_set_.build(
				candidates_, state_.collision_mesh, displaced_surface, dhat_);
		else
			constraint_set_.build(
				state_.collision_mesh, displaced_surface, dhat_, /*dmin=*/0, broad_phase_method_);
		cached_displaced_surface = displaced_surface;
	}

	double ContactForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return ipc::compute_barrier_potential(state_.collision_mesh, compute_displaced_surface(x), constraint_set_, dhat_);
	}

	void ContactForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = ipc::compute_barrier_potential_gradient(state_.collision_mesh, compute_displaced_surface(x), constraint_set_, dhat_);
		gradv = state_.collision_mesh.to_full_dof(gradv);
	}

	void ContactForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
	{
		POLYFEM_SCOPED_TIMER("\t\tbarrier hessian");
		hessian = ipc::compute_barrier_potential_hessian(state_.collision_mesh, compute_displaced_surface(x), constraint_set_, dhat_, project_to_psd_);
		hessian = state_.collision_mesh.to_full_dof(hessian);
	}

	void ContactForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		update_constraint_set(compute_displaced_surface(new_x));
	}

	double ContactForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// Extract surface only
		const Eigen::MatrixXd V0 = compute_displaced_surface(x0);
		const Eigen::MatrixXd V1 = compute_displaced_surface(x1);

		double max_step;
		if (use_cached_candidates_ && broad_phase_method_ != ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU)
			max_step = ipc::compute_collision_free_stepsize(
				candidates_, state_.collision_mesh, V0, V1, ccd_tolerance_, ccd_max_iterations_);
		else
			max_step = ipc::compute_collision_free_stepsize(
				state_.collision_mesh, V0, V1, broad_phase_method_, ccd_tolerance_, ccd_max_iterations_);

#ifndef NDEBUG
		// This will check for static intersections as a failsafe. Not needed if we use our conservative CCD.
		Eigen::MatrixXd V_toi = (V1 - V0) * max_step + V0;

		while (ipc::has_intersections(state_.collision_mesh, V_toi))
		{
			logger().error("taking max_step results in intersections (max_step={:g})", max_step);
			max_step /= 2.0;

			const double Linf = (V_toi - V0).lpNorm<Eigen::Infinity>();
			if (max_step <= 0 || Linf == 0)
				log_and_throw_error(fmt::format("Unable to find an intersection free step size (max_step={:g} L∞={:g})", max_step, Linf));

			V_toi = (V1 - V0) * max_step + V0;
		}
#endif

		return max_step;
	}

	void ContactForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		ipc::construct_collision_candidates(
			state_.collision_mesh,
			compute_displaced_surface(x0),
			compute_displaced_surface(x1),
			candidates_,
			/*inflation_radius=*/dhat_ / 1.99, // divide by 1.99 instead of 2 to be conservative
			broad_phase_method_);

		use_cached_candidates_ = true;
	}

	void ContactForm::line_search_end()
	{
		candidates_.clear();
		use_cached_candidates_ = false;
	}

	void ContactForm::post_step(const int iter_num, const Eigen::VectorXd &x)
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);

		const double curr_distance = ipc::compute_minimum_distance(state_.collision_mesh, displaced_surface, constraint_set_);

		if (use_adaptive_barrier_stiffness_)
		{
			if (is_time_dependent_)
			{
				const double prev_barrier_stiffness = weight_;

				weight_ = ipc::update_barrier_stiffness(
					prev_distance_, curr_distance, max_barrier_stiffness_,
					weight_, ipc::world_bbox_diagonal_length(displaced_surface));

				if (prev_barrier_stiffness != weight_)
				{
					polyfem::logger().debug(
						"updated barrier stiffness from {:g} to {:g}",
						prev_barrier_stiffness, weight_);
				}
			}
			else
			{
				// TODO: missing feature
				// update_barrier_stiffness(x);
			}
		}

		prev_distance_ = curr_distance;
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
			is_valid = ipc::is_step_collision_free(
				candidates_, state_.collision_mesh,
				displaced0,
				displaced1,
				ccd_tolerance_, ccd_max_iterations_);
		else
			is_valid = ipc::is_step_collision_free(
				state_.collision_mesh,
				displaced0,
				displaced1,
				broad_phase_method_, ccd_tolerance_, ccd_max_iterations_);

		return is_valid;
	}
} // namespace polyfem::solver

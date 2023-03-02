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
							 const bool use_convergent_formulation,
							 const bool use_adaptive_barrier_stiffness,
							 const bool is_time_dependent,
							 const ipc::BroadPhaseMethod broad_phase_method,
							 const double ccd_tolerance,
							 const int ccd_max_iterations)
		: collision_mesh_(collision_mesh),
		  dhat_(dhat),
		  avg_mass_(avg_mass),
		  use_adaptive_barrier_stiffness_(use_adaptive_barrier_stiffness),
		  is_time_dependent_(is_time_dependent),
		  broad_phase_method_(broad_phase_method),
		  ccd_tolerance_(ccd_tolerance),
		  ccd_max_iterations_(ccd_max_iterations)
	{
		assert(dhat_ > 0);
		assert(ccd_tolerance > 0);

		prev_distance_ = -1;
		constraint_set_.use_convergent_formulation = use_convergent_formulation;
	}

	void ContactForm::init(const Eigen::VectorXd &x)
	{
		update_constraint_set(compute_displaced_surface(x));
	}

	void ContactForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		update_constraint_set(compute_displaced_surface(x));
	}

	Eigen::MatrixXd ContactForm::compute_displaced_surface(const Eigen::VectorXd &x) const
	{
		return collision_mesh_.displace_vertices(utils::unflatten(x, collision_mesh_.dim()));
	}

	void ContactForm::update_barrier_stiffness(
		const Eigen::VectorXd &x,
		NLProblem &nl_problem,
		std::shared_ptr<FrictionForm> friction_form)
	{
		if (!use_adaptive_barrier_stiffness())
			return;

		const bool enabled_before = this->enabled();
		const bool friction_enabled_before = friction_form != nullptr && friction_form->enabled();

		this->disable();
		if (friction_form)
			friction_form->disable();

		Eigen::VectorXd grad_energy;
		nl_problem.gradient(x, grad_energy);

		this->set_enabled(enabled_before);
		if (friction_form)
			friction_form->set_enabled(friction_enabled_before);

		this->update_barrier_stiffness(x, grad_energy);
	}

	void ContactForm::update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy)
	{
		if (!use_adaptive_barrier_stiffness())
			return;

		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);

		Eigen::VectorXd grad_barrier = ipc::compute_barrier_potential_gradient(
			collision_mesh_, displaced_surface, constraint_set_, dhat_);
		grad_barrier = collision_mesh_.to_full_dof(grad_barrier);

		weight_ = ipc::initial_barrier_stiffness(
			ipc::world_bbox_diagonal_length(displaced_surface), dhat_, avg_mass_,
			grad_energy, grad_barrier, max_barrier_stiffness_);
		if (use_convergent_formulation())
		{
			// cancel out division in barrier potential
			weight_ *= dhat_ * dhat_;
			max_barrier_stiffness_ *= dhat_ * dhat_;
		}

		logger().debug("adaptive barrier form stiffness {}", barrier_stiffness());
	}

	void ContactForm::update_constraint_set(const Eigen::MatrixXd &displaced_surface)
	{
		// Store the previous value used to compute the constraint set to avoid duplicate computation.
		static Eigen::MatrixXd cached_displaced_surface;
		if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
			return;

		if (use_cached_candidates_)
			constraint_set_.build(
				candidates_, collision_mesh_, displaced_surface, dhat_);
		else
			constraint_set_.build(
				collision_mesh_, displaced_surface, dhat_, /*dmin=*/0, broad_phase_method_);
		cached_displaced_surface = displaced_surface;
	}

	double ContactForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return ipc::compute_barrier_potential(collision_mesh_, compute_displaced_surface(x), constraint_set_, dhat_);
	}

	void ContactForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = ipc::compute_barrier_potential_gradient(collision_mesh_, compute_displaced_surface(x), constraint_set_, dhat_);
		gradv = collision_mesh_.to_full_dof(gradv);
	}

	void ContactForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("barrier hessian");
		hessian = ipc::compute_barrier_potential_hessian(collision_mesh_, compute_displaced_surface(x), constraint_set_, dhat_, project_to_psd_);
		hessian = collision_mesh_.to_full_dof(hessian);
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

		if (save_ccd_debug_meshes)
		{
			const Eigen::MatrixXi E = collision_mesh_.dim() == 2 ? Eigen::MatrixXi() : collision_mesh_.edges();
			const Eigen::MatrixXi &F = collision_mesh_.faces();
			igl::writePLY(resolve_output_path("debug_ccd_0.ply"), V0, F, E);
			igl::writePLY(resolve_output_path("debug_ccd_1.ply"), V1, F, E);
		}

		double max_step;
		if (use_cached_candidates_ && broad_phase_method_ != ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU)
			max_step = ipc::compute_collision_free_stepsize(
				candidates_, collision_mesh_, V0, V1, ccd_tolerance_, ccd_max_iterations_);
		else
			max_step = ipc::compute_collision_free_stepsize(
				collision_mesh_, V0, V1, broad_phase_method_, ccd_tolerance_, ccd_max_iterations_);

#ifndef NDEBUG
		// This will check for static intersections as a failsafe. Not needed if we use our conservative CCD.
		Eigen::MatrixXd V_toi = (V1 - V0) * max_step + V0;

		while (ipc::has_intersections(collision_mesh_, V_toi))
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
			collision_mesh_,
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

		const double curr_distance = ipc::compute_minimum_distance(collision_mesh_, displaced_surface, constraint_set_);

		if (use_adaptive_barrier_stiffness_)
		{
			if (is_time_dependent_)
			{
				const double prev_barrier_stiffness = barrier_stiffness();

				weight_ = ipc::update_barrier_stiffness(
					prev_distance_, curr_distance, max_barrier_stiffness_,
					barrier_stiffness(), ipc::world_bbox_diagonal_length(displaced_surface));

				if (barrier_stiffness() != prev_barrier_stiffness)
				{
					polyfem::logger().debug(
						"updated barrier stiffness from {:g} to {:g}",
						prev_barrier_stiffness, barrier_stiffness());
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
				candidates_, collision_mesh_,
				displaced0,
				displaced1,
				ccd_tolerance_, ccd_max_iterations_);
		else
			is_valid = ipc::is_step_collision_free(
				collision_mesh_,
				displaced0,
				displaced1,
				broad_phase_method_, ccd_tolerance_, ccd_max_iterations_);

		return is_valid;
	}
} // namespace polyfem::solver

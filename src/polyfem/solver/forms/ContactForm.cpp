#pragma once

#include <polyfem/utils/Types.hpp>

namespace polyfem
{
	namespace solver
	{
		ContactForm::ContactForm()
		{
			use_adaptive_barrier_stiffness = !state.args["solver"]["contact"]["barrier_stiffness"].is_number();
			_dhat = dhat;
			assert(_dhat > 0);

			_broad_phase_method = state.args["solver"]["contact"]["CCD"]["broad_phase"];
			_ccd_tolerance = state.args["solver"]["contact"]["CCD"]["tolerance"];
			_ccd_max_iterations = state.args["solver"]["contact"]["CCD"]["max_iterations"];

			if (use_adaptive_barrier_stiffness)
			{
				_barrier_stiffness = 1;
				logger().debug("Using adaptive barrier stiffness");
			}
			else
			{
				assert(state.args["solver"]["contact"]["barrier_stiffness"].is_number());
				_barrier_stiffness = state.args["solver"]["contact"]["barrier_stiffness"];
				logger().debug("Using fixed barrier stiffness of {}", _barrier_stiffness);
			}

			_prev_distance = -1;
		}

		void ContactForm::update_barrier_stiffness(const TVector &full)
		{
			assert(full.size() == full_size);
			_barrier_stiffness = 1;

			Eigen::MatrixXd grad_energy; //TODO

			Eigen::MatrixXd displaced;
			compute_displaced_points(full, displaced);

			Eigen::MatrixXd displaced_surface = state.collision_mesh.vertices(displaced);
			update_constraint_set(displaced_surface);
			Eigen::VectorXd grad_barrier = ipc::compute_barrier_potential_gradient(
				state.collision_mesh, displaced_surface, _constraint_set, _dhat);
			grad_barrier = state.collision_mesh.to_full_dof(grad_barrier);

			_barrier_stiffness = ipc::initial_barrier_stiffness(
				ipc::world_bbox_diagonal_length(displaced), _dhat, state.avg_mass,
				grad_energy, grad_barrier, max_barrier_stiffness_);
			polyfem::logger().debug("adaptive barrier stiffness {}", _barrier_stiffness);
		}

		void ContactForm::update_constraint_set(const Eigen::MatrixXd &displaced_surface)
		{
			// Store the previous value used to compute the constraint set to avoid
			// duplicate computation.
			static Eigen::MatrixXd cached_displaced_surface;
			if (cached_displaced_surface.size() == displaced_surface.size()
				&& cached_displaced_surface == displaced_surface)
				return;

			if (_use_cached_candidates)
				ipc::construct_constraint_set(
					_candidates, state.collision_mesh, displaced_surface, _dhat,
					_constraint_set);
			else
				ipc::construct_constraint_set(
					state.collision_mesh, displaced_surface, _dhat,
					_constraint_set, /*dmin=*/0, _broad_phase_method);
			cached_displaced_surface = displaced_surface;
		}

		void ContactForm::init(const Eigen::VectorXd &displacement)
		{
			if (use_adaptive_barrier_stiffness)
			{
				update_barrier_stiffness(full);
			}
		}

		virtual double ContactForm::value(const Eigen::VectorXd &x)
		{
			return _barrier_stiffness * ipc::compute_barrier_potential(state.collision_mesh, displaced_surface, _constraint_set, _dhat);
		}
		virtual void ContactForm::gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
		{
			grad_barrier = _barrier_stiffness * ipc::compute_barrier_potential_gradient(state.collision_mesh, displaced_surface, _constraint_set, _dhat);
			grad_barrier = state.collision_mesh.to_full_dof(grad_barrier);
		}

		virtual void ContactForm::hessian(const Eigen::VectorXd &x, StiffnessMatrix &hessian)
		{
			POLYFEM_SCOPED_TIMER("\t\tbarrier hessian time");
			barrier_hessian = _barrier_stiffness * ipc::compute_barrier_potential_hessian(state.collision_mesh, displaced_surface, _constraint_set, _dhat, project_to_psd);
			barrier_hessian = state.collision_mesh.to_full_dof(barrier_hessian);
		}

		virtual void ContactForm::solution_changed(const Eigen::VectorXd &newX)
		{
			update_constraint_set(state.collision_mesh.vertices(displaced));
		}

		virtual double ContactForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
		{
			// Extract surface only
			Eigen::MatrixXd V0 = state.collision_mesh.vertices(displaced0);
			Eigen::MatrixXd V1 = state.collision_mesh.vertices(displaced1);

			// write_obj("s0.obj", V0, state.collision_mesh.edges(), state.collision_mesh.faces());
			// write_obj("s1.obj", V1, state.collision_mesh.edges(), state.collision_mesh.faces());

			double max_step;
			if (_use_cached_candidates
#ifdef IPC_TOOLKIT_WITH_CUDA
				&& _broad_phase_method != ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU
#endif
			)
				max_step = ipc::compute_collision_free_stepsize(
					_candidates, state.collision_mesh, V0, V1,
					_ccd_tolerance, _ccd_max_iterations);
			else
				max_step = ipc::compute_collision_free_stepsize(
					state.collision_mesh, V0, V1,
					_broad_phase_method, _ccd_tolerance, _ccd_max_iterations);
				// polyfem::logger().trace("best step {}", max_step);

#ifndef NDEBUG
			// This will check for static intersections as a failsafe. Not needed if we use our conservative CCD.
			Eigen::MatrixXd V_toi = (V1 - V0) * max_step + V0;
			while (ipc::has_intersections(state.collision_mesh, V_toi))
			{
				double Linf = (V_toi - V0).lpNorm<Eigen::Infinity>();
				logger().error("taking max_step results in intersections (max_step={:g})", max_step);
				max_step /= 2.0;
				if (max_step <= 0 || Linf == 0)
				{
					const std::string msg = fmt::format("Unable to find an intersection free step size (max_step={:g} L∞={:g})", max_step, Linf);
					logger().error(msg);
					throw std::runtime_error(msg);
				}
				V_toi = (V1 - V0) * max_step + V0;
			}
#endif

			return max_step;
		}

		virtual void ContactForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
		{
			ipc::construct_collision_candidates(
				state.collision_mesh,
				state.collision_mesh.vertices(displaced0),
				state.collision_mesh.vertices(displaced1),
				_candidates,
				/*inflation_radius=*/_dhat / 1.99, // divide by 1.99 instead of 2 to be conservative
				_broad_phase_method);

			_use_cached_candidates = true;
		}

		virtual void ContactForm::line_search_end()
		{
			_candidates.clear();
			_use_cached_candidates = false;
		}

		virtual void ContactForm::post_step(const int iter_num, const Eigen::VectorXd &x){}
		{
			Eigen::MatrixXd displaced_surface = state.collision_mesh.vertices(displaced);

			if (state.args["output"]["advanced"]["save_nl_solve_sequence"])
			{
				write_obj(state.resolve_output_path(fmt::format("step{:03d}.obj", iter_num)),
						  displaced_surface, state.collision_mesh.edges(), state.collision_mesh.faces());
			}
			if (use_adaptive_barrier_stiffness)
			{
				if (is_time_dependent)
				{
					double prev_barrier_stiffness = _barrier_stiffness;
					ipc::update_barrier_stiffness(
						_prev_distance, dist_sqr, max_barrier_stiffness_,
						_barrier_stiffness, ipc::world_bbox_diagonal_length(displaced_surface));
					if (prev_barrier_stiffness != _barrier_stiffness)
					{
						polyfem::logger().debug(
							"updated barrier stiffness from {:g} to {:g}",
							prev_barrier_stiffness, _barrier_stiffness);
					}
				}
				else
				{
					update_barrier_stiffness(full);
				}
			}
			_prev_distance = dist_sqr;
		}

		void ContactForm::update_quantities(const double t, const Eigen::VectorXd &x)
		{
			if (use_adaptive_barrier_stiffness)
			{
				update_barrier_stiffness(x);
			}
		}

	} // namespace solver
} // namespace polyfem

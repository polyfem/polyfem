#include "HighOrderContactForm.hpp"
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/io/OBJWriter.hpp>

#include <ipc/utils/eigen_ext.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>

namespace polyfem::solver
{
	ipc::HighOrderContactParameters init_params(const double dhat, const json &high_order_contact_params, int powerdefault, const bool skip_obstacles) {
		const int quadrature_order = high_order_contact_params["quadrature_order"];
		const double dbar_factor = high_order_contact_params["dbar_factor"];
		int power = high_order_contact_params["exponent"];
		if (power < 1) power = powerdefault;
		const ipc::HighOrderContactParameters::IntegrationType itype = skip_obstacles ?
			ipc::HighOrderContactParameters::IntegrationType::NO_OBST : ipc::HighOrderContactParameters::IntegrationType::NORMAL;
		return ipc::HighOrderContactParameters(dhat, dbar_factor, quadrature_order, power, itype);
	}

	HighOrderContactForm::HighOrderContactForm(const ipc::CollisionMesh &collision_mesh,
											   const double dhat,
											   const double avg_mass,
											   const json high_order_contact_params,
											   const bool skip_obstacles,
											   const bool use_adaptive_barrier_stiffness,
											   const bool is_time_dependent,
											   const bool enable_shape_derivatives,
											   const ipc::BroadPhaseMethod broad_phase_method,
											   const double ccd_tolerance,
											   const int ccd_max_iterations) : ContactForm(collision_mesh, dhat, avg_mass, use_adaptive_barrier_stiffness, is_time_dependent, enable_shape_derivatives, broad_phase_method, ccd_tolerance, ccd_max_iterations), params(init_params(dhat, high_order_contact_params, collision_mesh.dim() - 1, skip_obstacles)),
											   barrier_potential_(params, high_order_contact_params["normalize_weights"])
	{
	}

	void HighOrderContactForm::update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy)
	{
		if (!use_adaptive_barrier_stiffness())
			return;

		log_and_throw_error("Adaptive barrier stiffness not implemented for HighOrderContactForm!");
	}

	void HighOrderContactForm::force_shape_derivative(const ipc::HighOrderCollisions &collision_set, const Eigen::MatrixXd &solution, const Eigen::VectorXd &adjoint_sol, Eigen::VectorXd &term) const
	{
		StiffnessMatrix hessian = barrier_potential_.hessian(collision_set, collision_mesh_, compute_displaced_surface(solution), ipc::PSDProjectionMethod::NONE);
		term = barrier_stiffness() * collision_mesh_.to_full_dof(hessian) * adjoint_sol;
	}

	void HighOrderContactForm::update_collision_set(const Eigen::MatrixXd &displaced_surface)
	{
		// Store the previous value used to compute the constraint set to avoid duplicate computation.
		if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
			return;

		collision_set_.build(
			collision_mesh_, displaced_surface, params, /*use_adaptive_dhat*/ false, broad_phase_.get());
		cached_displaced_surface = displaced_surface;
	}

	double HighOrderContactForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd displaced = compute_displaced_surface(x);
		if (cached_displaced_surface != displaced) {
			return 0.;
		}
		return barrier_potential_(collision_set_, collision_mesh_, displaced);
	}

	Eigen::VectorXd HighOrderContactForm::value_per_element_unweighted(const Eigen::VectorXd &x) const
	{
		log_and_throw_error("value_per_element_unweighted not implemented!");
	}

	void HighOrderContactForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::MatrixXd displaced = compute_displaced_surface(x);
		if (cached_displaced_surface != displaced) {
			gradv.setZero(x.size());
			return;
		}
		gradv = barrier_potential_.gradient(collision_set_, collision_mesh_, displaced);
		gradv = collision_mesh_.to_full_dof(gradv);
	}

	void HighOrderContactForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		// {
		// 	static int hessian_call_count = 0;
		// 	const std::string filename = "collision_mesh_hessian_" + std::to_string(hessian_call_count++) + ".obj";
		// 	io::OBJWriter::write(
		// 		filename,
		// 		compute_displaced_surface(x),
		// 		collision_mesh_.edges(), collision_mesh_.faces());
		// 	polyfem::logger().debug("Exported collision mesh to {}", filename);
		// }
		POLYFEM_SCOPED_TIMER("barrier hessian");
		hessian = barrier_potential_.hessian(collision_set_, collision_mesh_, compute_displaced_surface(x), project_to_psd_ ? ipc::PSDProjectionMethod::CLAMP : ipc::PSDProjectionMethod::NONE);
		hessian = collision_mesh_.to_full_dof(hessian);
	}

	void HighOrderContactForm::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(data.x);

		// Always requires update_collision_set
		update_collision_set(displaced_surface);

		const double curr_distance = collision_set_.compute_minimum_distance(collision_mesh_, displaced_surface);
		if (!std::isinf(curr_distance))
		{
			const double ratio = sqrt(curr_distance) / dhat();
			const auto log_level = (ratio < 1e-6) ? spdlog::level::err : ((ratio < 1e-4) ? spdlog::level::warn : spdlog::level::debug);
			polyfem::logger().log(log_level, "Minimum distance during solve: {}, dhat: {}", sqrt(curr_distance), dhat());
		}

		if (data.iter_num == 0)
			return;

		if (use_adaptive_barrier_stiffness_)
		{
			if (is_time_dependent_)
			{
				const double prev_barrier_stiffness = barrier_stiffness();

				barrier_stiffness_ = ipc::update_barrier_stiffness(
					prev_distance_, curr_distance, max_barrier_stiffness_,
					barrier_stiffness(), ipc::world_bbox_diagonal_length(displaced_surface), 1e-7);

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
				// update_barrier_stiffness(data.x);
			}
		}

		prev_distance_ = curr_distance;
	}
} // namespace polyfem::solver
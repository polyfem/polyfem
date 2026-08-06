#include "TangentPointForm.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	TangentPointForm::TangentPointForm(const ipc::CollisionMesh &collision_mesh,
										const double dhat,
										const double avg_mass,
										const bool is_time_dependent,
										const bool enable_shape_derivatives,
										const ipc::BroadPhaseMethod broad_phase_method,
										const double ccd_tolerance,
										const int ccd_max_iterations,
										const double dhat_epsilon_scale)
		: ContactForm(collision_mesh, dhat, avg_mass, /*use_adaptive_barrier_stiffness=*/false,
					  is_time_dependent, enable_shape_derivatives, broad_phase_method,
					  ccd_tolerance, ccd_max_iterations, dhat_epsilon_scale),
		  tangent_point_potential_(dhat)
	{
	}

	double TangentPointForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return tangent_point_potential_(collision_mesh_, compute_displaced_surface(x));
	}

	void TangentPointForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = tangent_point_potential_.gradient(collision_mesh_, compute_displaced_surface(x));
		gradv = collision_mesh_.to_full_dof(gradv);
	}

	void TangentPointForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		ipc::PSDProjectionMethod psd_projection_method;

		if (project_to_psd_) {
			psd_projection_method = ipc::PSDProjectionMethod::CLAMP;
		} else {
			psd_projection_method = ipc::PSDProjectionMethod::NONE;
		}

		hessian = tangent_point_potential_.hessian(collision_mesh_, compute_displaced_surface(x), psd_projection_method);
		hessian = collision_mesh_.to_full_dof(hessian);
	}

	void TangentPointForm::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		const double curr_distance = tangent_point_potential_.minimum_distance(collision_mesh_, compute_displaced_surface(data.x));
		if (!std::isinf(curr_distance))
		{
			const double ratio = curr_distance / dhat();
			const auto log_level = (ratio < 1e-6) ? spdlog::level::err : ((ratio < 1e-4) ? spdlog::level::warn : spdlog::level::debug);
			polyfem::logger().log(log_level, "Minimum distance during solve: {}, dhat: {}", curr_distance, dhat());
		}
	}
} // namespace polyfem::solver

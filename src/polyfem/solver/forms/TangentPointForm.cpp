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
		// TPE's kernel has no built-in PSD guarantee (unlike the log-barrier
		// forms, whose Hessian is analytically well-behaved near the active
		// region). Unlike HighOrderContactForm's alternating +1/-1 weighted
		// sums, TPE's face pairs never carry opposing signs, so clamping each
		// pair's own Hessian (done inside TangentPointPotential::hessian) is
		// always sound — always request it rather than deferring to the
		// generic project_to_psd_ flag, which no forward-solve code path ever
		// sets to true.
		hessian = tangent_point_potential_.hessian(collision_mesh_, compute_displaced_surface(x), ipc::PSDProjectionMethod::CLAMP);
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

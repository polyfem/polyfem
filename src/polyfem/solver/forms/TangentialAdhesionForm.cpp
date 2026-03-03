#include "TangentialAdhesionForm.hpp"
#include "NormalAdhesionForm.hpp"

#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{
	TangentialAdhesionForm::TangentialAdhesionForm(
		const ipc::CollisionMesh &collision_mesh,
		const std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator,
		const double epsa,
		const double mu,
		const ipc::BroadPhaseMethod broad_phase_method,
		const NormalAdhesionForm &normal_adhesion_form,
		const int n_lagging_iters)
		: collision_mesh_(collision_mesh),
		  time_integrator_(time_integrator),
		  epsa_(epsa),
		  mu_(mu),
		  broad_phase_method_(broad_phase_method),
		  broad_phase_(ipc::create_broad_phase(broad_phase_method)),
		  n_lagging_iters_(n_lagging_iters < 0 ? std::numeric_limits<int>::max() : n_lagging_iters),
		  normal_adhesion_form_(normal_adhesion_form),
		  tangential_adhesion_potential_(epsa)
	{
		assert(epsa_ > 0);
	}

	Eigen::MatrixXd TangentialAdhesionForm::compute_displaced_surface(const Eigen::VectorXd &x) const
	{
		return normal_adhesion_form_.compute_displaced_surface(x);
	}

	Eigen::MatrixXd TangentialAdhesionForm::compute_surface_velocities(const Eigen::VectorXd &x) const
	{
		// In the case of a static problem, the velocity is the displacement
		const Eigen::VectorXd v = time_integrator_ != nullptr ? time_integrator_->compute_velocity(x) : x;
		return collision_mesh_.map_displacements(utils::unflatten(v, collision_mesh_.dim()));
	}

	double TangentialAdhesionForm::dv_dx() const
	{
		return time_integrator_ != nullptr ? time_integrator_->dv_dx() : 1;
	}

	double TangentialAdhesionForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return tangential_adhesion_potential_(tangential_collision_set_, collision_mesh_, compute_surface_velocities(x)) / dv_dx();
	}

	void TangentialAdhesionForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::VectorXd grad_tangential_adhesion = tangential_adhesion_potential_.gradient(
			tangential_collision_set_, collision_mesh_, compute_surface_velocities(x));
		gradv = collision_mesh_.to_full_dof(grad_tangential_adhesion);
	}

	void TangentialAdhesionForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("tangential adhesion hessian");

		ipc::PSDProjectionMethod psd_projection_method; 

		if (project_to_psd_) {
			psd_projection_method = ipc::PSDProjectionMethod::CLAMP;
		} else {
			psd_projection_method = ipc::PSDProjectionMethod::NONE;
		}

		hessian = dv_dx() * tangential_adhesion_potential_.hessian( //
					  tangential_collision_set_, collision_mesh_, compute_surface_velocities(x), psd_projection_method);

		hessian = collision_mesh_.to_full_dof(hessian);
	}

	void TangentialAdhesionForm::update_lagging(const Eigen::VectorXd &x, const int iter_num)
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);

		ipc::NormalCollisions collision_set;

		collision_set.build(
			collision_mesh_, displaced_surface, normal_adhesion_form_.dhat_a(), /*dmin=*/0, broad_phase_);

		tangential_collision_set_.build(
			collision_mesh_, displaced_surface, collision_set,
			normal_adhesion_form_.normal_adhesion_potential(),
			1., mu_);
	}
} // namespace polyfem::solver

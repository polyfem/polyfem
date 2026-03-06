#include "FrictionForm.hpp"
#include "BarrierContactForm.hpp"
#include "SmoothContactForm.hpp"

#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <ipc/broad_phase/create_broad_phase.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>
#include <ipc/smooth_contact/smooth_collisions.hpp>

#include <Eigen/Core>

#include <cassert>
#include <limits>
#include <memory>
#include <stdexcept>

namespace polyfem::solver
{
	FrictionForm::FrictionForm(
		const ipc::CollisionMesh &collision_mesh,
		const std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator,
		const double epsv,
		const double mu,
		const ipc::BroadPhaseMethod broad_phase_method,
		const ContactForm &contact_form,
		const int n_lagging_iters)
		: collision_mesh_(collision_mesh),
		  time_integrator_(time_integrator),
		  epsv_(epsv),
		  mu_(mu),
		  broad_phase_method_(broad_phase_method),
		  n_lagging_iters_(n_lagging_iters < 0 ? std::numeric_limits<int>::max() : n_lagging_iters),
		  contact_form_(contact_form),
		  friction_potential_(epsv)
	{
		assert(epsv_ > 0);
	}

	Eigen::MatrixXd FrictionForm::compute_displaced_surface(const Eigen::VectorXd &x) const
	{
		return contact_form_.compute_displaced_surface(x);
	}

	Eigen::MatrixXd FrictionForm::compute_surface_velocities(const Eigen::VectorXd &x) const
	{
		// In the case of a static problem, the velocity is the displacement
		const Eigen::VectorXd v = time_integrator_ != nullptr ? time_integrator_->compute_velocity(x) : x;
		return collision_mesh_.map_displacements(utils::unflatten(v, collision_mesh_.dim()));
	}

	double FrictionForm::dv_dx() const
	{
		return time_integrator_ != nullptr ? time_integrator_->dv_dx() : 1;
	}

	double FrictionForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return friction_potential_(friction_collision_set_, collision_mesh_, compute_surface_velocities(x)) / dv_dx();
	}

	void FrictionForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::VectorXd grad_friction = friction_potential_.gradient(
			friction_collision_set_, collision_mesh_, compute_surface_velocities(x));
		gradv = collision_mesh_.to_full_dof(grad_friction);
	}

	void FrictionForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("friction hessian");

		ipc::PSDProjectionMethod psd_projection_method;

		if (project_to_psd_)
		{
			psd_projection_method = ipc::PSDProjectionMethod::CLAMP;
		}
		else
		{
			psd_projection_method = ipc::PSDProjectionMethod::NONE;
		}

		hessian = dv_dx() * friction_potential_.hessian( //
					  friction_collision_set_, collision_mesh_, compute_surface_velocities(x), psd_projection_method);

		hessian = collision_mesh_.to_full_dof(hessian);
	}

	void FrictionForm::update_lagging(const Eigen::VectorXd &x, const int iter_num)
	{
		const Eigen::MatrixXd displaced_surface = compute_displaced_surface(x);

		auto broad_phase = ipc::create_broad_phase(broad_phase_method_);
		if (const auto barrier_contact = dynamic_cast<const BarrierContactForm *>(&contact_form_))
		{
			ipc::NormalCollisions collision_set;
			collision_set.set_use_area_weighting(barrier_contact->use_area_weighting());
			collision_set.set_use_improved_max_approximator(barrier_contact->use_improved_max_operator());

			collision_set.set_enable_shape_derivatives(barrier_contact->enable_shape_derivatives());
			collision_set.build(
				collision_mesh_, displaced_surface, barrier_contact->dhat(), /*dmin=*/0, broad_phase);

			friction_collision_set_.build(
				collision_mesh_, displaced_surface, collision_set,
				barrier_contact->barrier_potential(), barrier_contact->barrier_stiffness(), Eigen::VectorXd::Ones(collision_mesh_.num_vertices()) * mu_, Eigen::VectorXd::Ones(collision_mesh_.num_vertices()) * mu_);
		}
		else if (const auto smooth_contact = dynamic_cast<const SmoothContactForm *>(&contact_form_))
		{
			ipc::SmoothCollisions collision_set;
			if (smooth_contact->using_adaptive_dhat())
				collision_set.compute_adaptive_dhat(collision_mesh_, collision_mesh_.rest_positions(), smooth_contact->get_params(), broad_phase);
			collision_set.build(
				collision_mesh_, displaced_surface, smooth_contact->get_params(),
				smooth_contact->using_adaptive_dhat(), broad_phase);

			friction_collision_set_.build(
				collision_mesh_, displaced_surface,
				collision_set, smooth_contact->get_params(), contact_form_.barrier_stiffness(), Eigen::VectorXd::Ones(collision_mesh_.num_vertices()) * mu_, Eigen::VectorXd::Ones(collision_mesh_.num_vertices()) * mu_);
		}
		else
		{
			throw std::runtime_error("Unknown contact form");
		}
	}
} // namespace polyfem::solver

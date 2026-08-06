#pragma once

#include "ContactForm.hpp"

#include <polyfem/utils/Types.hpp>

#include <ipc/collision_mesh.hpp>
#include <ipc/potentials/tangent_point_potential.hpp>

namespace polyfem::solver
{
	/// @brief Form representing the (Repulsive Surfaces/Shells) tangent-point
	/// energy, a repulsive potential provided as an alternative contact
	/// formulation to BarrierContactForm/HighOrderContactForm/SmoothContactForm.
	/// It does NOT guarantee an intersection-free result on its own, and does
	/// not implement continuous collision detection (max_step_size/
	/// is_step_collision_free are unconstrained), unlike the other formulations.
	class TangentPointForm : public ContactForm
	{
	public:
		/// @brief Construct a new TangentPointForm object
		/// @param collision_mesh Reference to the collision mesh
		/// @param dhat Activation distance beyond which face pairs do not interact
		/// @param avg_mass Average mass of the mesh
		/// @param is_time_dependent Is the simulation time dependent?
		/// @param enable_shape_derivatives Enable shape derivatives computation
		/// @param broad_phase_method Broad phase method to use
		/// @param ccd_tolerance Continuous collision detection tolerance
		/// @param ccd_max_iterations Continuous collision detection maximum iterations
		/// @param dhat_epsilon_scale dhat_epsilon_scale used by ipc::update_barrier_stiffness
		TangentPointForm(const ipc::CollisionMesh &collision_mesh,
						  const double dhat,
						  const double avg_mass,
						  const bool is_time_dependent,
						  const bool enable_shape_derivatives,
						  const ipc::BroadPhaseMethod broad_phase_method,
						  const double ccd_tolerance,
						  const int ccd_max_iterations,
						  const double dhat_epsilon_scale);
		virtual ~TangentPointForm() = default;

		std::string name() const override { return "tangent_point"; }

		// -- No continuous collision detection ---------------------------------
		// TPE has no notion of a CCD-safe step (FaceFaceCandidate is not a
		// CollisionStencil, so ipc-toolkit's CCD machinery cannot be used here);
		// this matches the papers themselves, which do not guarantee
		// intersection-free steps.
		void init(const Eigen::VectorXd &x) override {}
		void update_quantities(const double t, const Eigen::VectorXd &x) override {}
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override { return 1.0; }
		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override {}
		void solution_changed(const Eigen::VectorXd &new_x) override {}
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override { return true; }

		/// @brief No-op: TPE has no adaptive barrier stiffness scheme, and
		/// use_adaptive_barrier_stiffness() is always false for this form, so
		/// SolveData::update_barrier_stiffness never actually calls this.
		void update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy) override {}

		/// @brief Log the minimum distance among currently active face pairs
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

	protected:
		/// @brief No-op: TPE recomputes its own candidate set internally on
		/// every energy/gradient/Hessian evaluation (see TangentPointPotential),
		/// and has no CCD-driven candidate cache to maintain.
		void update_collision_set(const Eigen::MatrixXd &displaced_surface) override {}

		/// @brief Compute the tangent-point potential value
		/// @param x Current solution
		/// @return Value of the tangent-point potential
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:
		const ipc::TangentPointPotential tangent_point_potential_;
	};
} // namespace polyfem::solver

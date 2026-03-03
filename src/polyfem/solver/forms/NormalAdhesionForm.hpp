#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>
#include <polysolve/nonlinear/PostStepData.hpp>

#include <ipc/broad_phase/broad_phase.hpp>
#include <ipc/broad_phase/create_broad_phase.hpp>
#include <ipc/candidates/candidates.hpp>
#include <ipc/ccd/tight_inclusion_ccd.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>
#include <ipc/collision_mesh.hpp>
#include <ipc/potentials/normal_adhesion_potential.hpp>

#include <memory>

namespace polyfem::solver
{
	/// @brief Form representing the contact potential and forces
	class NormalAdhesionForm : public Form
	{
		friend class NormalAdhesionForceDerivative;

	public:
		/// @brief Construct a new NormalAdhesion Form object
		/// @param collision_mesh Reference to the collision mesh
		/// @param dhat_p Distance of largest adhesion force
		/// @param dhat_a Adhesion activation distance
		/// @param Y Adhesion strength
		/// @param is_time_dependent Is the simulation time dependent?
		/// @param enable_shape_derivatives Enable shape derivatives
		/// @param broad_phase_method Broad phase method to use for distance and CCD evaluations
		/// @param ccd_tolerance Continuous collision detection tolerance
		/// @param ccd_max_iterations Continuous collision detection maximum iterations
		NormalAdhesionForm(const ipc::CollisionMesh &collision_mesh,
					const double dhat_p,
					const double dhat_a,
					const double Y,
					const bool is_time_dependent,
					const bool enable_shape_derivatives,
					const ipc::BroadPhaseMethod broad_phase_method,
					const double ccd_tolerance,
					const int ccd_max_iterations);
		virtual ~NormalAdhesionForm() = default;

		std::string name() const override { return "normal adhesion"; }

		/// @brief Initialize the form
		/// @param x Current solution
		void init(const Eigen::VectorXd &x) override;

	protected:
		/// @brief Compute the normal adhesion potential value
		/// @param x Current solution
		/// @return Value of the normal adhesion potential
		virtual double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the value of the form multiplied per element
		/// @param x Current solution
		/// @return Computed value
		Eigen::VectorXd value_per_element_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		virtual void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	public:
		/// @brief Update time-dependent fields
		/// @param t Current time
		/// @param x Current solution at time t
		void update_quantities(const double t, const Eigen::VectorXd &x) override;

		/// @brief Initialize variables used during the line search
		/// @param x0 Current solution
		/// @param x1 Next solution
		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		/// @brief Clear variables used during the line search
		void line_search_end() override;

		/// @brief Update cached fields upon a change in the solution
		/// @param new_x New solution
		void solution_changed(const Eigen::VectorXd &new_x) override;

		/// @brief Update fields after a step in the optimization
		/// @param iter_num Optimization iteration number
		/// @param x Current solution
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

		/// @brief Compute the displaced positions of the surface nodes
		Eigen::MatrixXd compute_displaced_surface(const Eigen::VectorXd &x) const;

		bool enable_shape_derivatives() const { return enable_shape_derivatives_; }

		/// @brief If true, output debug files
		bool save_ccd_debug_meshes = false;

		double dhat_a() const { return dhat_a_; }
		double dhat_p() const { return dhat_p_; }
		double Y() const { return Y_; }
		const ipc::NormalCollisions &collision_set() const { return collision_set_; }
		const ipc::NormalAdhesionPotential &normal_adhesion_potential() const { return normal_adhesion_potential_; }

	protected:
		/// @brief Update the cached candidate set for the current solution
		/// @param displaced_surface Vertex positions displaced by the current solution
		void update_collision_set(const Eigen::MatrixXd &displaced_surface);

		/// @brief Collision mesh
		const ipc::CollisionMesh &collision_mesh_;

		/// @brief Maximum adhesion strength distance
		const double dhat_p_;

		/// @brief Adhesion activation distance
		const double dhat_a_;

		/// @brief Adhesion strength
		const double Y_;

		/// @brief Minimum distance between elements
		const double dmin_ = 0;

		/// @brief Is the simulation time dependent?
		const bool is_time_dependent_;

		/// @brief Enable shape derivatives computation
		const bool enable_shape_derivatives_;

		/// @brief Broad phase method to use for distance and CCD evaluations
		const ipc::BroadPhaseMethod broad_phase_method_;
		const std::shared_ptr<ipc::BroadPhase> broad_phase_;
		/// @brief Continuous collision detection specification object
		const ipc::TightInclusionCCD tight_inclusion_ccd_;

		/// @brief Previous minimum distance between all elements
		double prev_distance_;

		/// @brief If true, use the cached candidate set for the current solution
		bool use_cached_candidates_ = false;
		/// @brief Cached constraint set for the current solution
		ipc::NormalCollisions collision_set_;
		/// @brief Cached candidate set for the current solution
		ipc::Candidates candidates_;

		const ipc::NormalAdhesionPotential normal_adhesion_potential_;
	};
} // namespace polyfem::solver

#pragma once

#include "Form.hpp"

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>

#include <ipc/ipc.hpp>
#include <ipc/collision_mesh.hpp>
#include <ipc/broad_phase/broad_phase.hpp>

// map BroadPhaseMethod values to JSON as strings
namespace ipc
{
	NLOHMANN_JSON_SERIALIZE_ENUM(
		ipc::BroadPhaseMethod,
		{{ipc::BroadPhaseMethod::HASH_GRID, "hash_grid"}, // also default
		 {ipc::BroadPhaseMethod::HASH_GRID, "HG"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "brute_force"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "BF"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "spatial_hash"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "SH"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "sweep_and_tiniest_queue"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "STQ"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "sweep_and_tiniest_queue_gpu"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "STQ_GPU"}})
} // namespace ipc

namespace polyfem::solver
{
	class NLProblem;
	class FrictionForm;

	/// @brief Form representing the contact potential and forces
	class ContactForm : public Form
	{
	public:
		/// @brief Construct a new Contact Form object
		/// @param collision_mesh Reference to the collision mesh
		/// @param dhat Barrier activation distance
		/// @param avg_mass Average mass of the mesh
		/// @param use_adaptive_barrier_stiffness If true, use an adaptive barrier stiffness
		/// @param is_time_dependent Is the simulation time dependent?
		/// @param broad_phase_method Broad phase method to use for distance and CCD evaluations
		/// @param ccd_tolerance Continuous collision detection tolerance
		/// @param ccd_max_iterations Continuous collision detection maximum iterations
		ContactForm(const ipc::CollisionMesh &collision_mesh,
					const double dhat,
					const double avg_mass,
					const bool use_convergent_formulation,
					const bool use_adaptive_barrier_stiffness,
					const bool is_time_dependent,
					const ipc::BroadPhaseMethod broad_phase_method,
					const double ccd_tolerance,
					const int ccd_max_iterations);

		/// @brief Initialize the form
		/// @param x Current solution
		void init(const Eigen::VectorXd &x) override;

	protected:
		/// @brief Compute the contact barrier potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	public:
		/// @brief Update time-dependent fields
		/// @param t Current time
		/// @param x Current solution at time t
		void update_quantities(const double t, const Eigen::VectorXd &x) override;

		/// @brief Determine the maximum step size allowable between the current and next solution
		/// @param x0 Current solution (step size = 0)
		/// @param x1 Next solution (step size = 1)
		/// @return Maximum allowable step size
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

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
		void post_step(const int iter_num, const Eigen::VectorXd &x) override;

		/// @brief returns the current barrier stiffness
		/// @return double the current barrier stifness
		double barrier_stiffness() const { return weight_; }

		/// @brief Checks if the step is collision free
		/// @return True if the step is collision free else false
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		/// @brief Update the barrier stiffness based on the current elasticity energy
		/// @param x Current solution
		/// @param nl_problem Nonlinear problem to use for computing the gradient
		/// @param friction_form Pointer to the friction form
		void update_barrier_stiffness(
			const Eigen::VectorXd &x,
			NLProblem &nl_problem,
			std::shared_ptr<FrictionForm> friction_form);

		/// @brief Update the barrier stiffness based on the current elasticity energy
		/// @param x Current solution
		void update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy);

		inline bool use_adaptive_barrier_stiffness() const { return use_adaptive_barrier_stiffness_; }
		inline bool use_convergent_formulation() const { return constraint_set_.use_convergent_formulation; }

		bool save_ccd_debug_meshes = false; ///< If true, output debug files

	private:
		const ipc::CollisionMesh &collision_mesh_;

		const double dhat_; ///< Barrier activation distance

		const double avg_mass_;

		const bool use_adaptive_barrier_stiffness_; ///< If true, use an adaptive barrier stiffness
		double max_barrier_stiffness_;              ///< Maximum barrier stiffness to use when using adaptive barrier stiffness

		const bool is_time_dependent_; ///< Is the simulation time dependent?

		const ipc::BroadPhaseMethod broad_phase_method_; ///< Broad phase method to use for distance and CCD evaluations
		const double ccd_tolerance_;                     ///< Continuous collision detection tolerance
		const int ccd_max_iterations_;                   ///< Continuous collision detection maximum iterations

		double prev_distance_; ///< Previous minimum distance between all elements

		bool use_cached_candidates_ = false; ///< If true, use the cached candidate set for the current solution
		ipc::Constraints constraint_set_;    ///< Cached constraint set for the current solution
		ipc::Candidates candidates_;         ///< Cached candidate set for the current solution

		/// @brief Compute the displaced positions of the surface nodes
		Eigen::MatrixXd compute_displaced_surface(const Eigen::VectorXd &x) const;

		/// @brief Update the cached candidate set for the current solution
		/// @param displaced_surface Vertex positions displaced by the current solution
		void update_constraint_set(const Eigen::MatrixXd &displaced_surface);
	};
} // namespace polyfem::solver

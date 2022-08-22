#pragma once

#include "Form.hpp"

#include <polyfem/State.hpp>
#include <polyfem/utils/Types.hpp>

#include <ipc/ipc.hpp>
#include <ipc/broad_phase/broad_phase.hpp>

namespace polyfem::solver
{
	/// @brief Form representing the contact potential and forces
	class ContactForm : public Form
	{
	public:
		/// @brief Construct a new Contact Form object
		/// @param state Reference to the simulation state
		/// @param dhat Barrier activation distance
		/// @param use_adaptive_barrier_stiffness If true, use an adaptive barrier stiffness
		/// @param barrier_stiffness Stiffness multiplier of the form
		/// @param is_time_dependent Is the simulation time dependent?
		/// @param broad_phase_method Broad phase method to use for distance and CCD evaluations
		/// @param ccd_tolerance Continuous collision detection tolerance
		/// @param ccd_max_iterations Continuous collision detection maximum iterations
		ContactForm(const State &state,
					const double dhat,
					const bool use_adaptive_barrier_stiffness,
					const double &barrier_stiffness,
					const bool is_time_dependent,
					const ipc::BroadPhaseMethod broad_phase_method,
					const double ccd_tolerance,
					const int ccd_max_iterations);

		/// @brief Initialize the form
		/// @param x Current solution
		void init(const Eigen::VectorXd &x) override;

		/// @brief Compute the contact barrier potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		double value(const Eigen::VectorXd &x) override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		void second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

		/// @brief Determine the maximum step size allowable between the current and next solution
		/// @param x0 Current solution (step size = 0)
		/// @param x1 Next solution (step size = 1)
		/// @return Maximum allowable step size
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

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

		/// @brief Update time-dependent fields
		/// @param t Current time
		/// @param x Current solution at time t
		void update_quantities(const double t, const Eigen::VectorXd &x) override;

	private:
		const State &state_; ///< Reference to the simulation state

		const double dhat_; ///< Barrier activation distance

		const bool use_adaptive_barrier_stiffness_; ///< If true, use an adaptive barrier stiffness
		const double &barrier_stiffness_;           ///< Stiffness multiplier of the form
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

		/// @brief Initialize the barrier stiffness based on the current elasticity energy
		/// @param x Current solution
		void initialize_barrier_stiffness(const Eigen::VectorXd &x);

		/// @brief Update the cached candidate set for the current solution
		/// @param displaced_surface Vertex positions displaced by the current solution
		void update_constraint_set(const Eigen::MatrixXd &displaced_surface);
	};
} // namespace polyfem::solver

#pragma once

#include "BarrierContactForm.hpp"

#include <Eigen/Core>
#include <polyfem/utils/Types.hpp>
#include <polysolve/nonlinear/PostStepData.hpp>

namespace polyfem::solver
{
  class PeriodicContactForceDerivative;

	/// @brief Form representing the contact potential and forces on a periodic mesh
	/// This form has a different input format of [fluctuation, affine], only can be used in NLHomoProblem
    class PeriodicContactForm : public BarrierContactForm
    {
    friend class PeriodicContactForceDerivative;

    public:
		/// @brief Construct a new Contact Form object
		/// @param periodic_collision_mesh 3x3 tiling of a periodic mesh
		/// @param tiled_to_single Index mapping from the tiled mesh to the original periodic mesh
		PeriodicContactForm(const ipc::CollisionMesh &periodic_collision_mesh,
                        const Eigen::VectorXi &tiled_to_single,
                        const double dhat,
                        const double avg_mass,
                        const bool use_area_weighting,
						const bool use_improved_max_operator,
						const bool use_physical_barrier,
                        const bool use_adaptive_barrier_stiffness,
                        const bool is_time_dependent,
						const bool enable_shape_derivatives,
                        const ipc::BroadPhaseMethod broad_phase_method,
                        const double ccd_tolerance,
                        const int ccd_max_iterations);

        void init(const Eigen::VectorXd &x) override;

        Eigen::VectorXd single_to_tiled(const Eigen::VectorXd &x) const;
        Eigen::VectorXd tiled_to_single_grad(const Eigen::VectorXd &grad) const;

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

		/// @brief Update cached fields upon a change in the solution
		/// @param new_x New solution
		void solution_changed(const Eigen::VectorXd &new_x) override;

		/// @brief Update fields after a step in the optimization
		/// @param iter_num Optimization iteration number
		/// @param x Current solution
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

		/// @brief Checks if the step is collision free
		/// @return True if the step is collision free else false
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const override;

		/// @brief Update the barrier stiffness based on the current elasticity energy
		/// @param x Current solution
		void update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy) override;

    private:
		void update_projection() const;

        const Eigen::VectorXi tiled_to_single_;
		const int n_single_dof_;
		mutable StiffnessMatrix proj;
    };
}

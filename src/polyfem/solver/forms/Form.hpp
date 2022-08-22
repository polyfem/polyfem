#pragma once

#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	class Form
	{
	public:
		virtual ~Form() {}

		/// @brief Initialize the form
		/// @param x Current solution
		virtual void init(const Eigen::VectorXd &x) {}

		/// @brief Compute the value of the form
		/// @param x Current solution
		/// @return Computed value
		virtual double value(const Eigen::VectorXd &x) const = 0;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		virtual void first_derivative(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const = 0;

		/// @brief Compute the second derivative of the value wrt x
		/// @note This is not marked const because ElasticForm needs to cache the matrix assembly.
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		virtual void second_derivative(const Eigen::VectorXd &x, StiffnessMatrix &hessian) = 0;

		/// @brief Determine if a step from solution x0 to solution x1 is allowed
		/// @param x0 Current solution
		/// @param x1 Proposed next solution
		/// @return True if the step is allowed
		virtual bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const { return true; }

		/// @brief Determine the maximum step size allowable between the current and next solution
		/// @param x0 Current solution (step size = 0)
		/// @param x1 Next solution (step size = 1)
		/// @return Maximum allowable step size
		virtual double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const { return 1; }

		/// @brief Initialize variables used during the line search
		/// @param x0 Current solution
		/// @param x1 Next solution
		virtual void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) {}

		/// @brief Clear variables used during the line search
		virtual void line_search_end() {}

		/// @brief Update fields after a step in the optimization
		/// @param iter_num Optimization iteration number
		/// @param x Current solution
		virtual void post_step(const int iter_num, const Eigen::VectorXd &x) {}

		/// @brief Update cached fields upon a change in the solution
		/// @param new_x New solution
		virtual void solution_changed(const Eigen::VectorXd &new_x) {}

		/// @brief Update time-dependent fields
		/// @param t Current time
		/// @param x Current solution at time t
		virtual void update_quantities(const double t, const Eigen::VectorXd &x) {}

		// TODO: more than one step

		/// @brief Initialize lagged fields
		/// @param x Current solution
		virtual void init_lagging(const Eigen::VectorXd &x){};

		/// @brief Update lagged fields
		/// @param x Current solution
		virtual void update_lagging(const Eigen::VectorXd &x){};

		/// @brief Set project to psd
		/// @param val If true, the form's second derivative is projected to be positive semidefinite
		void set_project_to_psd(bool val) { project_to_psd_ = val; }

		/// @brief Get if the form's second derivative is projected to psd
		bool is_project_to_psd() const { return project_to_psd_; }

	protected:
		bool project_to_psd_ = false; ///< If true, the form's second derivative is projected to be positive semidefinite
	};
} // namespace polyfem::solver

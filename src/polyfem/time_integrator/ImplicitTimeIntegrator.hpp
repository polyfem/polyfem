#pragma once

#include <polyfem/Common.hpp>

#include <Eigen/Core>

#include <map>
#include <vector>
#include <deque>

namespace polyfem::time_integrator
{
	/// Implicit time integrator of a second order ODE (equivently a system of coupled first order ODEs).
	class ImplicitTimeIntegrator
	{
	public:
		ImplicitTimeIntegrator() {}
		virtual ~ImplicitTimeIntegrator() = default;

		/// @brief Set the time integrator parameters from a json object.
		/// @param params json containing parameters specific to each time integrator
		virtual void set_parameters(const json &params) {}

		/// @brief Initialize the time integrator with the previous values for \f$x\f$, \f$v\f$, and \f$a\f$.
		/// @param x_prev previous value for the solution
		/// @param v_prev previous value for the velocity
		/// @param a_prev previous value for the acceleration
		/// @param dt time step size
		virtual void init(const Eigen::VectorXd &x_prev, const Eigen::VectorXd &v_prev, const Eigen::VectorXd &a_prev, double dt);

		/// @brief Update the time integration quantaties (i.e., \f$x\f$, \f$v\f$, and \f$a\f$).
		/// @param x new solution vector
		virtual void update_quantities(const Eigen::VectorXd &x) = 0;

		/// @brief Update the time integration quantaties (i.e., \f$x\f$, \f$v\f$, and \f$a\f$).
		/// @param x new solution vector
		/// @param quasistatic whether to update quasistaticly (i.e., without updating the velocity and acceleration)
		void update_quantities(const Eigen::VectorXd &x, const bool quasistatic);

		/// @brief Compute the predicted solution to be used in the inertia term \f$(x-\tilde{x})^TM(x-\tilde{x})\f$.
		/// @return value for \f$\tilde{x}\f$
		virtual Eigen::VectorXd x_tilde() const = 0;

		/// @brief Compute the current velocity given the current solution and using the stored previous solution(s).
		/// @param x current solution
		/// @return value for \f$v\f$
		virtual Eigen::VectorXd compute_velocity(const Eigen::VectorXd &x) const = 0;

		/// @brief Compute the current acceleration given the current velocity and using the stored previous velocity(s).
		/// @param v current velocity
		/// @return value for \f$a\f$
		virtual Eigen::VectorXd compute_acceleration(const Eigen::VectorXd &v) const = 0;

		/// @brief Compute the acceleration scaling used to scale forces when integrating a second order ODE.
		/// @return value of the acceleration scaling
		virtual double acceleration_scaling() const = 0;

		/// @brief Access the time step size.
		const double &dt() const { return _dt; }

		/// @brief Save the values of \$x\$, \f$v\f$, and \f$a\f$.
		/// @param x_path path for the output file containing \f$x\f$, if the extension is `.txt`
		///               then it will write an ASCII file else if the extension is `.bin` it will
		///               write a binary file.
		/// @param v_path same as `x_path`, but for saving \f$v\f$
		/// @param a_path same as `x_path`, but for saving \f$a\f$
		virtual void save_raw(const std::string &x_path, const std::string &v_path, const std::string &a_path) const;

		/// @brief Factory method for constructing implicit time integrators from the name of the integrator.
		/// @param name name of the type of ImplicitTimeIntegrator to construct
		/// @return new implicit time integrator of type specfied by name
		static std::shared_ptr<ImplicitTimeIntegrator> construct_time_integrator(const json &params);

		/// @brief Get a vector of the names of possible ImplicitTimeIntegrators
		/// @return names in no particular order
		static const std::vector<std::string> &get_time_integrator_names();

		/// Get the most recent previous solution value.
		const Eigen::VectorXd &x_prev() const { return x_prevs.front(); }
		/// Get the most recent previous velocity value.
		const Eigen::VectorXd &v_prev() const { return v_prevs.front(); }
		/// Get the most recent previous acceleration value.
		const Eigen::VectorXd &a_prev() const { return a_prevs.front(); }

	protected:
		/// @brief Time step size.
		/// Default of one for static sims, this should be set using init().
		double _dt = 1;

		/// Store the necessary previous values of the solution for single or multi-step integration.
		std::deque<Eigen::VectorXd> x_prevs;
		/// Store the necessary previous values of the velocity for single or multi-step integration.
		std::deque<Eigen::VectorXd> v_prevs;
		/// Store the necessary previous values of the acceleration for single or multi-step integration.
		std::deque<Eigen::VectorXd> a_prevs;

		/// Convenience functions for setting the most recent previous solution.
		void set_x_prev(const Eigen::VectorXd &x_prev) { x_prevs.front() = x_prev; }
		/// Convenience functions for setting the most recent previous velocity.
		void set_v_prev(const Eigen::VectorXd &v_prev) { v_prevs.front() = v_prev; }
		/// Convenience functions for setting the most recent previous acceleration.
		void set_a_prev(const Eigen::VectorXd &a_prev) { a_prevs.front() = a_prev; }
	};
} // namespace polyfem::time_integrator

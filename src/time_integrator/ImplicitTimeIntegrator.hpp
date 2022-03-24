#pragma once

#include <map>
#include <vector>
#include <deque>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

namespace polyfem
{

	class ImplicitTimeIntegrator
	{
	public:
		ImplicitTimeIntegrator() {}
		virtual ~ImplicitTimeIntegrator() = default;

		virtual void set_parameters(const nlohmann::json &params) {}

		virtual void init(const Eigen::VectorXd &x_prev, const Eigen::VectorXd &v_prev, const Eigen::VectorXd &a_prev, double dt);

		virtual void update_quantities(const Eigen::VectorXd &x) = 0;

		virtual Eigen::VectorXd x_tilde() const = 0;

		virtual double acceleration_scaling() const = 0;

		const double &dt() const { return _dt; }

		virtual void save_raw(const std::string &x_path, const std::string &v_path, const std::string &a_path) const;

		static std::shared_ptr<ImplicitTimeIntegrator> construct_time_integrator(const std::string &name);
		static const std::vector<std::string> &get_time_integrator_names();

		// Convenience functions for getting the most recent previous values
		const Eigen::VectorXd &x_prev() const { return x_prevs.front(); }
		const Eigen::VectorXd &v_prev() const { return v_prevs.front(); }
		const Eigen::VectorXd &a_prev() const { return a_prevs.front(); }

	protected:
		double _dt;
		// Store the necessary previous values for single or multi-step integration
		std::deque<Eigen::VectorXd> x_prevs, v_prevs, a_prevs;

		// Convenience functions for getting the most recent previous values
		Eigen::VectorXd &x_prev() { return x_prevs.front(); }
		Eigen::VectorXd &v_prev() { return v_prevs.front(); }
		Eigen::VectorXd &a_prev() { return a_prevs.front(); }
	};
} // namespace polyfem

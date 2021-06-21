#pragma once

#include <map>
#include <vector>

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

	protected:
		double _dt;
		Eigen::VectorXd x_prev, v_prev, a_prev;
	};

} // namespace polyfem

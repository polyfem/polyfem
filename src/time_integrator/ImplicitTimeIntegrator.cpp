#include <polyfem/ImplicitTimeIntegrator.hpp>

#include <fstream>

#include <polyfem/ImplicitEuler.hpp>
#include <polyfem/ImplicitNewmark.hpp>
#include <polyfem/Logger.hpp>

namespace polyfem
{

	void ImplicitTimeIntegrator::init(const Eigen::VectorXd &x_prev, const Eigen::VectorXd &v_prev, const Eigen::VectorXd &a_prev, double dt)
	{
		this->x_prev = x_prev;
		this->v_prev = v_prev;
		this->a_prev = a_prev;
		_dt = dt;
	}

	void ImplicitTimeIntegrator::save_raw(const std::string &x_path, const std::string &v_path, const std::string &a_path) const
	{
		if (!x_path.empty())
		{
			std::ofstream os(x_path);
			os << x_prev;
		}

		if (!v_path.empty())
		{
			std::ofstream os(v_path);
			os << v_prev;
		}

		if (!a_path.empty())
		{
			std::ofstream os(a_path);
			os << a_prev;
		}
	}

	std::shared_ptr<ImplicitTimeIntegrator> ImplicitTimeIntegrator::construct_time_integrator(const std::string &name)
	{
		if (name == "ImplicitEuler")
		{
			return std::make_shared<ImplicitEuler>();
		}
		else if (name == "ImplicitNewmark")
		{
			return std::make_shared<ImplicitNewmark>();
		}
		else
		{
			logger().warn("Unknown time integrator ({}). Using implicit Euler.", name);
			return std::make_shared<ImplicitEuler>();
		}
	}

	const std::vector<std::string> &ImplicitTimeIntegrator::get_time_integrator_names()
	{
		static const std::vector<std::string> names = {{"ImplicitEuler", "ImplicitNewmark"}};
		return names;
	}

} // namespace polyfem

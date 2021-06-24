#include <polyfem/ImplicitTimeIntegrator.hpp>

#include <polyfem/ImplicitEuler.hpp>
#include <polyfem/ImplicitNewmark.hpp>
#include <polyfem/Logger.hpp>
#include <polyfem/MatrixUtils.hpp>

#include <fstream>

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
			write_matrix_binary(x_path, x_prev);

		if (!v_path.empty())
			write_matrix_binary(v_path, v_prev);

		if (!a_path.empty())
			write_matrix_binary(a_path, a_prev);
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
		static const std::vector<std::string> names = {{std::string("ImplicitEuler"), std::string("ImplicitNewmark")}};
		return names;
	}

} // namespace polyfem

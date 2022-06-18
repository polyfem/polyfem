#include "ImplicitTimeIntegrator.hpp"

#include <polyfem/time_integrator/ImplicitEuler.hpp>
#include <polyfem/time_integrator/ImplicitNewmark.hpp>
#include <polyfem/time_integrator/BDF.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <fstream>

namespace polyfem
{
	using namespace utils;

	namespace time_integrator
	{
		void ImplicitTimeIntegrator::init(const Eigen::VectorXd &x_prev, const Eigen::VectorXd &v_prev, const Eigen::VectorXd &a_prev, double dt)
		{
			x_prevs.clear();
			x_prevs.push_front(x_prev);

			v_prevs.clear();
			v_prevs.push_front(v_prev);

			a_prevs.clear();
			a_prevs.push_front(a_prev);

			assert(dt > 0);
			_dt = dt;
		}

		void ImplicitTimeIntegrator::save_raw(const std::string &x_path, const std::string &v_path, const std::string &a_path) const
		{
			if (!x_path.empty())
				write_matrix(x_path, x_prev());

			if (!v_path.empty())
				write_matrix(v_path, v_prev());

			if (!a_path.empty())
				write_matrix(a_path, a_prev());
		}

		std::shared_ptr<ImplicitTimeIntegrator> ImplicitTimeIntegrator::construct_time_integrator(const std::string &name)
		{
			if (name == "implict_euler" || name == "ImplicitEuler")
			{
				return std::make_shared<ImplicitEuler>();
			}
			else if (name == "implict_newmark" || name == "ImplicitNewmark")
			{
				return std::make_shared<ImplicitNewmark>();
			}
			else if (name == "BDF")
			{
				return std::make_shared<BDF>();
			}
			else
			{
				logger().warn("Unknown time integrator ({}); using implicit Euler", name);
				return std::make_shared<ImplicitEuler>();
			}
		}

		const std::vector<std::string> &ImplicitTimeIntegrator::get_time_integrator_names()
		{
			static const std::vector<std::string> names = {
				std::string("ImplicitEuler"),
				std::string("ImplicitNewmark"),
				std::string("BDF"),
			};
			return names;
		}
	} // namespace time_integrator
} // namespace polyfem

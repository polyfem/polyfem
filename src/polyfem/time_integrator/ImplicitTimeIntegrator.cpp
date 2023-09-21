#include "ImplicitTimeIntegrator.hpp"

#include <polyfem/time_integrator/ImplicitEuler.hpp>
#include <polyfem/time_integrator/ImplicitNewmark.hpp>
#include <polyfem/time_integrator/BDF.hpp>

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <fstream>

namespace polyfem
{
	using namespace io;
	namespace time_integrator
	{
		void ImplicitTimeIntegrator::init(const Eigen::VectorXd &x_prev, const Eigen::VectorXd &v_prev, const Eigen::VectorXd &a_prev, double dt)
		{
			x_prevs_.clear();
			x_prevs_.push_front(x_prev);

			v_prevs_.clear();
			v_prevs_.push_front(v_prev);

			a_prevs_.clear();
			a_prevs_.push_front(a_prev);

			assert(dt > 0);
			dt_ = dt;
		}

		void ImplicitTimeIntegrator::init(
			const std::vector<Eigen::VectorXd> &x_prevs,
			const std::vector<Eigen::VectorXd> &v_prevs,
			const std::vector<Eigen::VectorXd> &a_prevs,
			double dt)
		{
			assert(x_prevs.size() > 0 && x_prevs.size() <= max_steps());
			assert(x_prevs.size() == v_prevs.size());
			assert(x_prevs.size() == a_prevs.size());

			x_prevs_.clear();
			v_prevs_.clear();
			a_prevs_.clear();

			const int n = std::min(int(x_prevs.size()), max_steps());
			for (int i = 0; i < n; i++)
			{
				x_prevs_.push_back(x_prevs[i]);
				v_prevs_.push_back(v_prevs[i]);
				a_prevs_.push_back(a_prevs[i]);
			}

			assert(dt > 0);
			dt_ = dt;
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

		std::shared_ptr<ImplicitTimeIntegrator> ImplicitTimeIntegrator::construct_time_integrator(const json &params)
		{
			const std::string type = params.is_object() ? params["type"] : params;

			std::shared_ptr<ImplicitTimeIntegrator> integrator;
			if (type == "implict_euler" || type == "ImplicitEuler")
			{
				integrator = std::make_shared<ImplicitEuler>();
			}
			else if (type == "implict_newmark" || type == "ImplicitNewmark")
			{
				integrator = std::make_shared<ImplicitNewmark>();
			}
			else if (utils::StringUtils::startswith(type, "BDF"))
			{
				integrator = std::make_shared<BDF>(type == "BDF" ? 1 : std::stoi(type.substr(3)));
			}
			else
			{
				logger().error("Unknown time integrator ({})", type);
				throw std::runtime_error(fmt::format("Unknown time integrator ({})", type));
			}

			if (params.is_object())
				integrator->set_parameters(params);

			return integrator;
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

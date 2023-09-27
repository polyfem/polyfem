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
		void ImplicitTimeIntegrator::init(
			const Eigen::MatrixXd &x_prevs,
			const Eigen::MatrixXd &v_prevs,
			const Eigen::MatrixXd &a_prevs,
			double dt)
		{
			assert(x_prevs.cols() > 0 && x_prevs.cols() <= max_steps());
			assert(x_prevs.cols() == v_prevs.cols());
			assert(x_prevs.cols() == a_prevs.cols());

			x_prevs_.clear();
			v_prevs_.clear();
			a_prevs_.clear();

			const int n = std::min(int(x_prevs.cols()), max_steps());
			for (int i = 0; i < n; i++)
			{
				x_prevs_.push_back(x_prevs.col(i));
				v_prevs_.push_back(v_prevs.col(i));
				a_prevs_.push_back(a_prevs.col(i));
			}

			assert(dt > 0);
			dt_ = dt;
		}

		void ImplicitTimeIntegrator::save_state(const std::string &state_path) const
		{
			assert(!state_path.empty());

			const int ndof = x_prev().size();
			const int prev_steps = x_prevs().size();

			Eigen::MatrixXd tmp(ndof, prev_steps);

			for (int i = 0; i < prev_steps; ++i)
				tmp.col(i) = x_prevs()[i];
			write_matrix(state_path, "u", tmp, /*replace=*/true);

			for (int i = 0; i < prev_steps; ++i)
				tmp.col(i) = v_prevs()[i];
			write_matrix(state_path, "v", tmp, /*replace=*/false);

			for (int i = 0; i < prev_steps; ++i)
				tmp.col(i) = a_prevs()[i];
			write_matrix(state_path, "a", tmp, /*replace=*/false);
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

#include "ImplicitTimeIntegrator.hpp"

#include <polyfem/time_integrator/ImplicitEuler.hpp>
#include <polyfem/time_integrator/ImplicitNewmark.hpp>
#include <polyfem/time_integrator/BDF.hpp>

#include <polyfem/io/Checkpoint.hpp>
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
			const int ndof = x_prev().size();
			const int history = x_prevs().size();
			Eigen::MatrixXd values(ndof, history);
			for (int i = 0; i < history; ++i)
				values.col(i) = x_prevs()[i];
			write_matrix(state_path, "u", values, true);
			for (int i = 0; i < history; ++i)
				values.col(i) = v_prevs()[i];
			write_matrix(state_path, "v", values, false);
			for (int i = 0; i < history; ++i)
				values.col(i) = a_prevs()[i];
			write_matrix(state_path, "a", values, false);
		}

		void ImplicitTimeIntegrator::serialize_checkpoint(io::CheckpointWriter &writer, const std::string &group) const
		{
			if (x_prevs_.empty() || x_prevs_.size() != v_prevs_.size() || x_prevs_.size() != a_prevs_.size())
				log_and_throw_error("Cannot checkpoint an uninitialized or inconsistent time integrator.");
			const int ndof = x_prevs_.front().size();
			const int history = x_prevs_.size();
			Eigen::MatrixXd x(ndof, history), v(ndof, history), a(ndof, history);
			for (int i = 0; i < history; ++i)
			{
				if (x_prevs_[i].size() != ndof || v_prevs_[i].size() != ndof || a_prevs_[i].size() != ndof)
					log_and_throw_error("Cannot checkpoint inconsistent time-integrator history dimensions.");
				x.col(i) = x_prevs_[i];
				v.col(i) = v_prevs_[i];
				a.col(i) = a_prevs_[i];
			}
			writer.write_matrix(group + "/x", x);
			writer.write_matrix(group + "/v", v);
			writer.write_matrix(group + "/a", a);
			writer.write_long(group + "/dynamic_order", dynamic_order_ == DynamicOrder::First ? 1 : 2);
			writer.write_long(group + "/history_length", history);
			writer.write_double(group + "/dt", dt_);
		}

		void ImplicitTimeIntegrator::deserialize_checkpoint(
			const io::CheckpointReader &reader,
			const std::string &group,
			const double expected_dt)
		{
			for (const std::string &key : {"x", "v", "a", "dynamic_order", "history_length", "dt"})
				if (!reader.exists(group + "/" + key))
					log_and_throw_error("Checkpoint integrator group {} is missing {}.", group, key);
			const long order = reader.read_long(group + "/dynamic_order");
			if (order != (dynamic_order_ == DynamicOrder::First ? 1 : 2))
				log_and_throw_error("Checkpoint dynamic order is incompatible with {}.", group);
			const long history = reader.read_long(group + "/history_length");
			if (history < 1 || history > max_steps())
				log_and_throw_error("Checkpoint history length {} is incompatible with {}.", history, group);
			const double stored_dt = reader.read_double(group + "/dt");
			if (std::abs(stored_dt - expected_dt) > 1e-12 * std::max({1.0, std::abs(stored_dt), std::abs(expected_dt)}))
				log_and_throw_error("Checkpoint dt {} does not match configured dt {}.", stored_dt, expected_dt);
			const Eigen::MatrixXd x = reader.read_matrix(group + "/x");
			const Eigen::MatrixXd v = reader.read_matrix(group + "/v");
			const Eigen::MatrixXd a = reader.read_matrix(group + "/a");
			if (x.rows() == 0 || x.cols() != history || v.rows() != x.rows() || a.rows() != x.rows()
				|| v.cols() != history || a.cols() != history)
				log_and_throw_error("Checkpoint integrator history dimensions are invalid in {}.", group);
			init(x, v, a, stored_dt);
		}

		std::shared_ptr<ImplicitTimeIntegrator> ImplicitTimeIntegrator::construct_time_integrator(
			const json &params,
			const DynamicOrder dynamic_order)
		{
			const std::string type = params.is_object() ? params["type"] : params;

			std::shared_ptr<ImplicitTimeIntegrator> integrator;
			if (type == "implict_euler" || type == "ImplicitEuler")
			{
				integrator = std::make_shared<ImplicitEuler>(dynamic_order);
			}
			else if (type == "implict_newmark" || type == "ImplicitNewmark")
			{
				integrator = std::make_shared<ImplicitNewmark>(dynamic_order);
			}
			else if (utils::StringUtils::startswith(type, "BDF"))
			{
				integrator = std::make_shared<BDF>(type == "BDF" ? 1 : std::stoi(type.substr(3)), dynamic_order);
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

		std::shared_ptr<BDF> ImplicitTimeIntegrator::construct_bdf_integrator(
			const json &params,
			const DynamicOrder dynamic_order)
		{
			const std::string type = params.is_object() ? params["type"] : params;
			if (type != "implict_euler" && type != "ImplicitEuler"
				&& !utils::StringUtils::startswith(type, "BDF"))
			{
				log_and_throw_error(
					"BDF-specific transient formulations require ImplicitEuler or BDF, got {}.",
					type);
			}

			auto integrator = std::make_shared<BDF>(
				utils::StringUtils::startswith(type, "BDF") && type != "BDF"
					? std::stoi(type.substr(3))
					: 1,
				dynamic_order);
			if (params.is_object() && params.contains("steps"))
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

#include "Logger.hpp"
#include <polyfem/utils/DisableWarnings.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <polyfem/utils/EnableWarnings.hpp>

#include <sstream>

namespace polyfem
{
	namespace
	{

		// Custom logger instance defined by the user, if any
		std::shared_ptr<spdlog::logger> &get_shared_logger()
		{
			static std::shared_ptr<spdlog::logger> logger;
			return logger;
		}

		// Custom logger instance defined by the user, if any
		std::shared_ptr<spdlog::logger> &get_shared_adjoint_logger()
		{
			static std::shared_ptr<spdlog::logger> logger;
			return logger;
		}

	} // namespace

	// Retrieve current logger
	spdlog::logger &adjoint_logger()
	{
		if (get_shared_adjoint_logger())
		{
			return *get_shared_adjoint_logger();
		}
		else
		{
			static std::shared_ptr<spdlog::logger> default_logger = spdlog::stdout_color_mt("adjoint-polyfem");
			return *default_logger;
		}
	}

	// Retrieve current logger
	spdlog::logger &logger()
	{
		if (get_shared_logger())
		{
			return *get_shared_logger();
		}
		else
		{
			// When using factory methods provided by spdlog (_st and _mt functions),
			// names must be unique, since the logger is registered globally.
			// Otherwise, you will need to create the logger manually. See
			// https://github.com/gabime/spdlog/wiki/2.-Creating-loggers
			static std::shared_ptr<spdlog::logger> default_logger = spdlog::stdout_color_mt("polyfem");
			return *default_logger;
		}
	}

	// Use a custom logger
	void set_logger(std::shared_ptr<spdlog::logger> p_logger)
	{
		get_shared_logger() = std::move(p_logger);
	}

	// Use a custom logger
	void set_adjoint_logger(std::shared_ptr<spdlog::logger> p_logger)
	{
		get_shared_adjoint_logger() = std::move(p_logger);
	}

	void log_and_throw_error(const std::string &msg)
	{
		logger().error(msg);
		throw std::runtime_error(msg);
	}

	void log_and_throw_adjoint_error(const std::string &msg)
	{
		adjoint_logger().error(msg);
		throw std::runtime_error(msg);
	}
} // namespace polyfem

fmt::format_context::iterator fmt::formatter<polyfem::StiffnessMatrix>::format(polyfem::StiffnessMatrix const &mat, fmt::format_context &ctx) const
{
	std::stringstream ss;
	ss << mat;
	return formatter<fmt::string_view>::format(ss.str(), ctx);
}

fmt::format_context::iterator fmt::formatter<Eigen::MatrixXd>::format(const Eigen::MatrixXd &mat, fmt::format_context &ctx) const
{
	std::stringstream ss;
	ss << mat;
	return formatter<fmt::string_view>::format(ss.str(), ctx);
}
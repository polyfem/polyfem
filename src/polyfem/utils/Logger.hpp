#pragma once

#include <polyfem/utils/DisableWarnings.hpp>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <polyfem/utils/EnableWarnings.hpp>

#include <polyfem/utils/Types.hpp>

namespace polyfem
{
	///
	/// Retrieves the current logger.
	///
	/// @return     A const reference to Polyfem's logger object.
	///
	spdlog::logger &logger();
	///
	/// Retrieves the current logger for adjoint.
	///
	/// @return     A const reference to Polyfem's logger object.
	///
	spdlog::logger &adjoint_logger();

	///
	/// Setup a logger object to be used by Polyfem. Calling this function with other Polyfem function
	/// is not thread-safe.
	///
	/// @param[in]  logger  New logger object to be used by Polyfem. Ownership is shared with Polyfem.
	///
	void set_logger(std::shared_ptr<spdlog::logger> logger);

	///
	/// Setup a logger object to be used by adjoint Polyfem. Calling this function with other Polyfem function
	/// is not thread-safe.
	///
	/// @param[in]  logger  New logger object to be used by adjoint Polyfem. Ownership is shared with Polyfem.
	///
	void set_adjoint_logger(std::shared_ptr<spdlog::logger> logger);

	[[noreturn]] void log_and_throw_error(const std::string &msg);
	[[noreturn]] void log_and_throw_adjoint_error(const std::string &msg);

	template <typename... Args>
	[[noreturn]] void log_and_throw_error(const std::string &msg, const Args &...args)
	{
		log_and_throw_error(fmt::format(msg, args...));
	}

	template <typename... Args>
	[[noreturn]] void log_and_throw_adjoint_error(const std::string &msg, const Args &...args)
	{
		log_and_throw_error(fmt::format(msg, args...));
	}
} // namespace polyfem

template <>
struct fmt::formatter<polyfem::StiffnessMatrix> : fmt::formatter<fmt::string_view>
{
	format_context::iterator format(const polyfem::StiffnessMatrix &mat, fmt::format_context &ctx) const;
};

template <>
struct fmt::formatter<Eigen::MatrixXd> : fmt::formatter<fmt::string_view>
{
	format_context::iterator format(const Eigen::MatrixXd &mat, fmt::format_context &ctx) const;
};

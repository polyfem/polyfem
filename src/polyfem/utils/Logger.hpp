#pragma once

#include <polyfem/utils/DisableWarnings.hpp>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <polyfem/utils/EnableWarnings.hpp>

namespace polyfem
{
	///
	/// Retrieves the current logger.
	///
	/// @return     A const reference to Polyfem's logger object.
	///
	spdlog::logger &logger();

	///
	/// Setup a logger object to be used by Polyfem. Calling this function with other Polyfem function
	/// is not thread-safe.
	///
	/// @param[in]  logger  New logger object to be used by Polyfem. Ownership is shared with Polyfem.
	///
	void set_logger(std::shared_ptr<spdlog::logger> logger);

	[[noreturn]] void log_and_throw_error(const std::string &msg);

	template <typename... Args>
	[[noreturn]] void log_and_throw_error(const std::string &msg, const Args &...args)
	{
		log_and_throw_error(fmt::format(msg, args...));
	}
} // namespace polyfem

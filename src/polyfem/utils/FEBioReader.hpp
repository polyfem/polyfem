#pragma once

#include <polyfem/State.hpp>

#include <string>

namespace polyfem
{
	namespace utils
	{
		class FEBioReader
		{
		public:
			static void load(const std::string &path, const json &args_in, State &state, const std::string &export_solution = "");
		};
	} // namespace utils
} // namespace polyfem

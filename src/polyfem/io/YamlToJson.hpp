#pragma once

#include <polyfem/Common.hpp>

namespace polyfem::io
{
	/// @brief Convert YAML string to JSON.
	json yaml_string_to_json(const std::string &yaml_str);
} // namespace polyfem::io

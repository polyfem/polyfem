#pragma once

#include <polyfem/Common.hpp>

namespace polyfem::io
{
	/// @brief Convert YAML string to JSON.
	json yaml_string_to_json(const std::string &yaml_str);

	/// @brief Load a YAML file to JSON.
	json yaml_file_to_json(const std::string &yaml_filepath);
} // namespace polyfem::io

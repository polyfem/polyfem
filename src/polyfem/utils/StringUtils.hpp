#pragma once

#include <vector>
#include <string>

namespace polyfem
{
	namespace utils
	{
		namespace StringUtils
		{

			// Split a string into tokens
			std::vector<std::string> split(const std::string &str, const std::string &delimiters = " ");

			// Skip comments in a stream
			std::istream &skip(std::istream &in, char x = '#');

			// Tests whether a string starts with a given prefix
			bool startswith(const std::string &str, const std::string &prefix);

			// Tests whether a string ends with a given suffix
			bool endswith(const std::string &str, const std::string &suffix);

			// Replace extension after the last "dot"
			std::string replace_ext(const std::string &filename, const std::string &newext);

			// Trims a string
			std::string trim(const std::string &string);

		} // namespace StringUtils

		std::string resolve_path(
			const std::string &path,
			const std::string &input_file_path,
			const bool only_if_exists = false);
	} // namespace utils
} // namespace polyfem

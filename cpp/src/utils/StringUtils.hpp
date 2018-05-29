#pragma once

#include <vector>
#include <string>

namespace poly_fem {

	namespace StringUtils {

		// Split a string into tokens
		std::vector<std::string> split(const std::string &str, const std::string &delimiters = " ");

		// Skip comments in a stream
		std::istream &skip(std::istream &in, char x = '#');

		// Tests whether a string starts with a given prefix
		bool startswith(const std::string &str, const std::string &prefix);

		// Tests whether a string ends with a given suffix
		bool endswidth(const std::string &str, const std::string &suffix);

		// Replace extension after the last "dot"
		std::string replace_ext(const std::string &filename, const std::string &newext);

	}

} // namespace poly_fem

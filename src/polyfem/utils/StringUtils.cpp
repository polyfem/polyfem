#include "StringUtils.hpp"
#include <iomanip>
#include <algorithm>
#include <functional>

#include <filesystem>

// Split a string into tokens
std::vector<std::string> polyfem::utils::StringUtils::split(const std::string &str, const std::string &delimiters)
{
	// Skip delimiters at beginning.
	std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	// Find first "non-delimiter".
	std::string::size_type pos = str.find_first_of(delimiters, lastPos);

	std::vector<std::string> tokens;
	while (std::string::npos != pos || std::string::npos != lastPos)
	{
		// Found a token, add it to the vector.
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		// Skip delimiters.  Note the "not_of"
		lastPos = str.find_first_not_of(delimiters, pos);
		// Find next "non-delimiter"
		pos = str.find_first_of(delimiters, lastPos);
	}

	return tokens;
}

// Skip comments in a stream
std::istream &polyfem::utils::StringUtils::skip(std::istream &in, char x)
{
	std::string dummy;
	while ((in >> std::ws).peek() == std::char_traits<char>::to_int_type(x))
	{
		std::getline(in, dummy);
	}
	return in;
}

// Tests whether a string starts with a given prefix
bool polyfem::utils::StringUtils::startswith(const std::string &str, const std::string &prefix)
{
	return (str.compare(0, prefix.size(), prefix) == 0);
}

// Tests whether a string ends with a given suffix
bool polyfem::utils::StringUtils::endswith(const std::string &str, const std::string &suffix)
{
	if (str.length() >= suffix.length())
	{
		return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
	}
	else
	{
		return false;
	}
}

// Replace extension after the last "dot"
std::string polyfem::utils::StringUtils::replace_ext(const std::string &filename, const std::string &newext)
{
	std::string ext = "";
	if (!newext.empty())
	{
		ext = (newext[0] == '.' ? newext : "." + newext);
	}
	size_t lastdot = filename.find_last_of(".");
	if (lastdot == std::string::npos)
	{
		return filename + ext;
	}
	return filename.substr(0, lastdot) + ext;
}

namespace
{
	// trim from start (in place)
	inline std::string ltrim(const std::string &s)
	{
		static const std::string WHITESPACE = " \n\r\t";

		size_t startpos = s.find_first_not_of(WHITESPACE);
		return (startpos == std::string::npos) ? "" : s.substr(startpos);
	}

	// trim from end (in place)
	inline std::string rtrim(const std::string &s)
	{
		static const std::string WHITESPACE = " \n\r\t";

		size_t endpos = s.find_last_not_of(WHITESPACE);
		return (endpos == std::string::npos) ? "" : s.substr(0, endpos + 1);
	}
} // namespace

// trim from both ends (copying)
std::string polyfem::utils::StringUtils::trim(const std::string &string)
{
	return rtrim(ltrim(string));
}

std::string polyfem::utils::resolve_path(
	const std::string &path,
	const std::string &input_file_path,
	const bool only_if_exists)
{
	if (path.empty())
	{
		return path;
	}

	std::filesystem::path resolved_path(path);
	if (resolved_path.is_absolute())
	{
		return resolved_path.string();
	}
	else if (std::filesystem::exists(resolved_path))
	{
		return std::filesystem::weakly_canonical(resolved_path).string();
	}

	std::filesystem::path input_dir_path(input_file_path);
	if (!std::filesystem::is_directory(input_dir_path))
		input_dir_path = input_dir_path.parent_path();

	resolved_path = std::filesystem::weakly_canonical(input_dir_path / resolved_path);

	if (only_if_exists && !std::filesystem::exists(resolved_path))
	{
		return path; // return path unchanged
	}
	return resolved_path.string();
}

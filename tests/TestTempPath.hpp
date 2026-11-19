#pragma once

#include <filesystem>
#include <random>
#include <string>

namespace polyfem::test
{
	/// A path in the temporary directory that is unique to this run of the
	/// suite, so that runs from several build trees at once do not share (and
	/// delete) each other's files.
	inline std::filesystem::path unique_temp_path(const std::string &stem, const std::string &extension = "")
	{
		static const std::string suffix = std::to_string(std::random_device{}());
		return std::filesystem::temp_directory_path() / (stem + "-" + suffix + extension);
	}
} // namespace polyfem::test

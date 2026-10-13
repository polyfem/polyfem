#pragma once

#include <polyfem/io/ResourceIO.hpp>

namespace polyfem::io
{
	class LoadedInput
	{
	public:
		json config;
		std::unique_ptr<const ResourceIO> resources;
	};

	LoadedInput load_json_input(const std::filesystem::path &path);
	LoadedInput load_yaml_input(const std::filesystem::path &path);
	LoadedInput load_hdf5_input(const std::filesystem::path &path);
} // namespace polyfem::io

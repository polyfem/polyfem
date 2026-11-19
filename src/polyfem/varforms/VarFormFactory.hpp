#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/io/ResourceIO.hpp>

#include <memory>
#include <string>

namespace polyfem::varform
{
	class VarForm;

	class VarFormFactory
	{
	public:
		static bool supports(const std::string &formulation, const json &args, bool is_optimization = false);
		static std::shared_ptr<VarForm> create(
			const std::string &formulation,
			const json &args,
			const io::ResourceIO &resources,
			bool is_optimization = false);
	};
} // namespace polyfem::varform

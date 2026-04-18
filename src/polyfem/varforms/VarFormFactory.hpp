#pragma once

#include <polyfem/Common.hpp>

#include <memory>
#include <string>

namespace polyfem::varform
{
	class VarForm;

	class VarFormFactory
	{
	public:
		static bool supports(const std::string &formulation, const json &args);
		static std::shared_ptr<VarForm> create(const std::string &formulation, const json &args);
	};
} // namespace polyfem::varform

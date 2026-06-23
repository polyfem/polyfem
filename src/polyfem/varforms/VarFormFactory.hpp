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

	/// @brief Extracts the formulation type from the given JSON arguments. this is temporary until legacy state is removed
	/// @param args JSON arguments containing material information.
	/// @return The formulation type as a string.
	std::string formulation_from_args(const json &args);

	/// @brief Checks if the given JSON arguments use a VarForm state. this is temporary until legacy state is removed
	/// @param args JSON arguments containing material information.
	/// @return True if the JSON arguments use a VarForm state, false otherwise.
	bool uses_varform_state(json args);
} // namespace polyfem::varform

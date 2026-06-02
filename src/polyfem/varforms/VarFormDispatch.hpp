#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/varforms/VarFormFactory.hpp>

namespace polyfem::varform
{
	inline std::string formulation_from_args(const json &args)
	{
		if (!args.contains("materials") || args["materials"].is_null())
			return "";

		if (args["materials"].is_array())
		{
			std::string current;
			for (const auto &m : args["materials"])
			{
				const std::string tmp = m["type"];
				if (current.empty())
					current = tmp;
				else if (current != tmp)
				{
					if (assembler::AssemblerUtils::is_elastic_material(current)
						&& assembler::AssemblerUtils::is_elastic_material(tmp))
					{
						current = "MultiModels";
					}
					else
					{
						return "";
					}
				}
			}

			return current;
		}

		return args["materials"].value("type", "");
	}

	inline bool uses_varform_state(json args)
	{
		utils::apply_common_params(args);
		const std::string formulation = formulation_from_args(args);
		return !formulation.empty() && VarFormFactory::create(formulation, args) != nullptr;
	}
} // namespace polyfem::varform

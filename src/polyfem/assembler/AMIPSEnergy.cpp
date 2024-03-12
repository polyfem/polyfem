#include "AMIPSEnergy.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/JSONUtils.hpp>

namespace polyfem::assembler
{

	AMIPSEnergy::AMIPSEnergy()
	{
		canonical_transformation_.resize(0);
	}

	std::map<std::string, Assembler::ParamFunc> AMIPSEnergy::parameters() const
	{
		return std::map<std::string, ParamFunc>();
	}

	void AMIPSEnergy::add_multimaterial(const int index, const json &params, const Units &)
	{
		if (params.contains("canonical_transformation"))
		{
			canonical_transformation_.reserve(params["canonical_transformation"].size());
			for (int i = 0; i < params["canonical_transformation"].size(); ++i)
			{
				canonical_transformation_.push_back(params["canonical_transformation"][i]);
			}
		}
	}

} // namespace polyfem::assembler
#include "AMIPSEnergy.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::assembler
{
	void AMIPSEnergy::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);

		if (params.contains("use_rest_pose"))
		{
			use_rest_pose_ = params["use_rest_pose"].get<bool>();
		}
	}

} // namespace polyfem::assembler
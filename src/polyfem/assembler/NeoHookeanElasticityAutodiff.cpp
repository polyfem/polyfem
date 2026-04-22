#include "NeoHookeanElasticityAutodiff.hpp"

DECLARE_DIFFSCALAR_BASE();

namespace polyfem::assembler
{
	NeoHookeanAutodiff::NeoHookeanAutodiff()
	{
	}

	void NeoHookeanAutodiff::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		params_.add_multimaterial(index, params, size() == 3, units.stress(), root_path);
	}
} // namespace polyfem::assembler
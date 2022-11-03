#include "NeoHookeanElasticityAutodiff.hpp"

#include <polyfem/basis/Basis.hpp>

namespace polyfem::assembler
{
	NeoHookeanAutodiff::NeoHookeanAutodiff()
	{
	}

	void NeoHookeanAutodiff::add_multimaterial(const int index, const json &params, const int size)
	{
		params_.add_multimaterial(index, params, size == 3);
	}
} // namespace polyfem::assembler
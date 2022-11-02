#include "MooneyRivlinElasticity.hpp"

#include <polyfem/basis/Basis.hpp>

namespace polyfem::assembler
{
	MooneyRivlinElasticity::MooneyRivlinElasticity()
		: c1_("c1"), c2_("c2"), k_("k")
	{
	}

	void MooneyRivlinElasticity::add_multimaterial(const int index, const json &params, const int size)
	{
		c1_.add_multimaterial(index, params);
		c2_.add_multimaterial(index, params);
		k_.add_multimaterial(index, params);
	}
} // namespace polyfem::assembler
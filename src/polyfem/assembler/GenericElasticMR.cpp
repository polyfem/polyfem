#include "MooneyRivlinElasticity.hpp"
#include "MooneyRivlin3ParamElasticity.hpp"

#include "GenericElastic.cpp"

namespace polyfem::assembler
{
	template class GenericElastic<MooneyRivlinElasticity>;
	template class GenericElastic<MooneyRivlin3ParamElasticity>;
} // namespace polyfem::assembler

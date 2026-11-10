#include "OgdenElasticity.hpp"

#include "GenericElastic.cpp"

namespace polyfem::assembler
{
	template class GenericElastic<UnconstrainedOgdenElasticity>;
	template class GenericElastic<IncompressibleOgdenElasticity>;
} // namespace polyfem::assembler

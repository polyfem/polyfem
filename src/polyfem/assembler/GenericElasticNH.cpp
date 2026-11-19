#include "NeoHookeanElasticityAutodiff.hpp"
#include "IsochoricNeoHookean.hpp"

#include "GenericElastic.cpp"

namespace polyfem::assembler
{
	template class GenericElastic<NeoHookeanAutodiff>;
	template class GenericElastic<IsochoricNeoHookean>;
} // namespace polyfem::assembler

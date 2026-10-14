#include "VolumePenalty.hpp"
#include "AMIPSEnergy.hpp"

#include "GenericElastic.cpp"

namespace polyfem::assembler
{
	template class GenericElastic<VolumePenalty>;
	template class GenericElastic<AMIPSEnergy>;
} // namespace polyfem::assembler

#include "HGOFiber.hpp"
#include "ActiveFiber.hpp"
#include "HGODispersion.hpp"

#include "GenericElastic.cpp"

namespace polyfem::assembler
{
	template class GenericElastic<HGOFiber>;
	template class GenericElastic<ActiveFiber>;
	template class GenericElastic<HGODispersion>;
} // namespace polyfem::assembler

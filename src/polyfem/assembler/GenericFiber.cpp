#include "GenericFiber.hpp"

#include <polyfem/assembler/HGOFiber.hpp>

namespace polyfem::assembler
{
	template <typename FiberModel>
	GenericFiber<FiberModel>::GenericFiber()
	{
	}

	template <typename FiberModel>
	void GenericFiber<FiberModel>::add_multimaterial(const int index, const json &params, const Units &units)
	{
		if (params.contains("fiber_direction"))
			fiber_direction_.add_multimaterial(index, params["fiber_direction"], units.length());
	}

	template <typename FiberModel>
	void GenericFiber<FiberModel>::set_size(const int size)
	{
		GenericElastic<FiberModel>::set_size(size);

		fiber_direction_.resize(size);
	}

	template class GenericFiber<HGOFiber>;
} // namespace polyfem::assembler
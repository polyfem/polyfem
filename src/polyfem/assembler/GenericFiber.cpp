#include "GenericFiber.hpp"

namespace polyfem::assembler
{
	GenericFiber::GenericFiber()
	{
	}

	void GenericFiber::add_multimaterial(const int index, const json &params, const Units &units)
	{
		assert(size() == 2 || size() == 3);

		if (params.contains("fiber_direction"))
			fiber_direction_.add_multimaterial(index, params["fiber_direction"], units.length());
	}

	void GenericFiber::set_size(const int size)
	{
		Assembler::set_size(size);
		fiber_direction_.resize(size);
	}
} // namespace polyfem::assembler
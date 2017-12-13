#ifndef ELEMENT_BASES_HPP
#define ELEMENT_BASES_HPP

#include "Basis.hpp"
#include "Quadrature.hpp"

#include <vector>

namespace poly_fem
{
	class ElementBases
	{
	public:
		std::vector<Basis> bases;

		Quadrature quadrature;

		bool has_parameterization = true;
	};
}

#endif

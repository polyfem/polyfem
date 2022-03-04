#pragma once

#include <polyfem/LineSearch.hpp>

#include <cppoptlib/linesearch/morethuente.h>

namespace polyfem
{
	template <typename ProblemType>
	class MoreThuenteLineSearch : public LineSearch<ProblemType>
	{
	public:
		using Superclass = LineSearch<ProblemType>;
		using typename Superclass::Scalar;
		using typename Superclass::TVector;

		double line_search(
			const TVector &x,
			const TVector &searchDir,
			ProblemType &objFunc) override
		{
			return cppoptlib::MoreThuente<ProblemType, 1>::linesearch(x, searchDir, objFunc);
		}
	};
} // namespace polyfem

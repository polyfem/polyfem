#pragma once

#include <polyfem/utils/Types.hpp>

#include "autodiff.h"

namespace polyfem
{

	typedef DScalar1<double, VectorNd> AutodiffScalarGrad;
	typedef DScalar2<double, VectorNd, MatrixNd> AutodiffScalarHessian;

	typedef VectorN<AutodiffScalarGrad> AutodiffGradPt;
	typedef VectorN<AutodiffScalarHessian> AutodiffHessianPt;

	// typedef DScalar1<double, VectorNd> AutodiffPt;

	template <class T>
	class AutoDiffAllocator
	{
	public:
		T operator()(const int i, double v) const
		{
			return T(i, v);
		}
	};

	template <>
	class AutoDiffAllocator<double>
	{
	public:
		double operator()(const int i, double v) const
		{
			return v;
		}
	};
} // namespace polyfem

#pragma once

#include <polyfem/utils/Types.hpp>

#include "autodiff.h"

namespace polyfem
{

	typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>> AutodiffScalarGrad;
	typedef DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>> AutodiffScalarHessian;

	typedef Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> AutodiffGradPt;
	typedef Eigen::Matrix<AutodiffScalarHessian, Eigen::Dynamic, 1, 0, 3, 1> AutodiffHessianPt;

	// typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>> 					AutodiffPt;

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

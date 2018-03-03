#pragma once

#include "autodiff.h"
#include <Eigen/Dense>

namespace poly_fem
{
	// Stack-allocated vectors of size either 2 or 3
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> VectorNd;
	typedef Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor, 1, 3> RowVectorNd;


	typedef Eigen::VectorXd						Gradient;
	typedef Eigen::MatrixXd						Hessian;
	typedef DScalar1<double, Gradient> 			AutoDiffScalar1;
	typedef DScalar2<double, Gradient, Hessian> AutoDiffScalar2;

	template<typename T>
	class AutoDiffAllocator
	{
	public:
		T operator()(const int i, double v) const
		{
			assert(false);
		}
	};

	template<>
	class AutoDiffAllocator<double>
	{
	public:
		double operator()(const int i, double v) const
		{
			return v;
		}
	};

	template<>
	class AutoDiffAllocator<AutoDiffScalar1>
	{
	public:
		AutoDiffScalar1 operator()(const int i, double v) const
		{
			return AutoDiffScalar1(i, v);
		}
	};

	template<>
	class AutoDiffAllocator<AutoDiffScalar2>
	{
	public:
		AutoDiffScalar2 operator()(const int i, double v) const
		{
			return AutoDiffScalar2(i, v);
		}
	};
}


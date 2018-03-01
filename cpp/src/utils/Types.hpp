#pragma once

#include "autodiff.h"
#include <Eigen/Dense>

namespace poly_fem
{
	// Stack-allocated vectors of size either 2 or 3
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> VectorNd;
	typedef Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor, 1, 3> RowVectorNd;


	typedef Eigen::VectorXd															Gradient;
	typedef Eigen::MatrixXd															Hessian;
	typedef DScalar2<double, Gradient, Hessian> 									AutoDiffScalar;
	typedef Eigen::Matrix<AutoDiffScalar, Eigen::Dynamic, 1> 						AutoDiffVect;
	typedef Eigen::Matrix<AutoDiffScalar, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> 	AutoDiffGradMat;
}
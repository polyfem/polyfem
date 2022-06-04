#pragma once

#include <polyfem/utils/ElasticityUtils.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <Eigen/Dense>

namespace polyfem
{
	namespace autogen
	{
		void linear_elasticity_2d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res);
		void linear_elasticity_3d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res);

		void hooke_2d_function(const AutodiffHessianPt &pt, const ElasticityTensor &C, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res);
		void hooke_3d_function(const AutodiffHessianPt &pt, const ElasticityTensor &C, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res);

		void saint_venant_2d_function(const AutodiffHessianPt &pt, const ElasticityTensor &C, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res);
		void saint_venant_3d_function(const AutodiffHessianPt &pt, const ElasticityTensor &C, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res);

		void neo_hookean_2d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res);
		void neo_hookean_3d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> &res);

	} // namespace autogen
} // namespace polyfem

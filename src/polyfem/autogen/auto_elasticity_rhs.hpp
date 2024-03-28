#pragma once

#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <Eigen/Dense>

namespace polyfem::autogen
{
	void linear_elasticity_2d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, VectorNd &res);
	void linear_elasticity_3d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, VectorNd &res);

	void hooke_2d_function(const AutodiffHessianPt &pt, const assembler::ElasticityTensor &C, VectorNd &res);
	void hooke_3d_function(const AutodiffHessianPt &pt, const assembler::ElasticityTensor &C, VectorNd &res);

	void saint_venant_2d_function(const AutodiffHessianPt &pt, const assembler::ElasticityTensor &C, VectorNd &res);
	void saint_venant_3d_function(const AutodiffHessianPt &pt, const assembler::ElasticityTensor &C, VectorNd &res);

	void neo_hookean_2d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, VectorNd &res);
	void neo_hookean_3d_function(const AutodiffHessianPt &pt, const double lambda, const double mu, VectorNd &res);
} // namespace polyfem::autogen

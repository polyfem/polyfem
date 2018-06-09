#pragma once

#include "Common.hpp"

#include "ElementAssemblyValues.hpp"
#include "ElementBases.hpp"

#include "AutodiffTypes.hpp"

#include <Eigen/Dense>


namespace poly_fem
{
	class LinearElasticity
	{
	public:
		// res is R^{m x dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da);

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void compute_von_mises_stresses(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		void compute_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;

		inline int &size() { return size_; }
		inline int size() const { return size_; }

		inline double &mu() { return mu_; }
		inline double mu() const { return mu_; }

		inline double &lambda() { return lambda_; }
		inline double lambda() const { return lambda_; }

		void set_parameters(const json &params);
	private:
		int size_ = 2;
		double mu_ = 1;
		double lambda_ = 1;
	};
}

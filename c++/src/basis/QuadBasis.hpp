#ifndef QUAD_BASIS_HPP
#define QUAD_BASIS_HPP

#include "Basis.hpp"

#include <Eigen/Dense>

namespace poly_fem
{
	class QuadBasis : public Basis
	{
	public:
		QuadBasis(const int global_index, const Eigen::MatrixXd &coeff, const int disc_order = 1);

		void basis(const Eigen::MatrixXd &uv, const int index, Eigen::MatrixXd &val) const override;
		void grad(const Eigen::MatrixXd &uv, const int index, Eigen::MatrixXd &val) const override;
		// void trasform(const Eigen::MatrixXd &uv, const Eigen::MatrixXd &quad, Eigen::MatrixXd &pts) const;
	private:
		int disc_order_;
	};
}

#endif //QUAD_BASIS_HPP

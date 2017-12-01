#ifndef HEX_BASIS_HPP
#define HEX_BASIS_HPP

#include "Basis.hpp"

#include <Eigen/Dense>

namespace poly_fem
{
	class HexBasis : public Basis
	{
	public:
		HexBasis(const int global_index, const Eigen::MatrixXd &coeff, const int disc_order = 1);

		void basis(const Eigen::MatrixXd &uv, const int index, Eigen::MatrixXd &val) const override;
		void grad(const Eigen::MatrixXd &uv, const int index, Eigen::MatrixXd &val) const override;
	private:
		int disc_order_;
	};
}

#endif //HEX_BASIS_HPP

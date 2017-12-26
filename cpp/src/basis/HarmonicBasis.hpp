#ifndef HARMONIC_BASIS_HPP
#define HARMONIC_BASIS_HPP

#include <Eigen/Dense>

namespace poly_fem
{
	class HarmonicBasis
	{
	public:
		HarmonicBasis(const Eigen::MatrixXd &centers, const Eigen::MatrixXd &samples, const Eigen::MatrixXd &rhs);

		void basis(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;
		void grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const;
	private:
		void compute(const Eigen::MatrixXd &samples, const Eigen::MatrixXd &rhs);

		Eigen::MatrixXd centers_;
		Eigen::MatrixXd weights_;

	};
}

#endif

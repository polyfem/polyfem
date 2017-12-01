#ifndef BASIS_HPP
#define BASIS_HPP

#include <Eigen/Dense>

namespace poly_fem
{
	class Basis
	{
	public:
		Basis(const int global_index, const Eigen::MatrixXd &coeff)
		: global_index_(global_index), coeff_(coeff)
		{ }

		virtual ~Basis() { }

		virtual void basis(const Eigen::MatrixXd &uv, const int index, Eigen::MatrixXd &val) const = 0;
		virtual void grad(const Eigen::MatrixXd &uv, const int index, Eigen::MatrixXd &val) const = 0;

		inline int global_index() const { return global_index_; }
		inline const Eigen::MatrixXd &coeff() const { return coeff_; }

	private:
		int global_index_;
		Eigen::MatrixXd coeff_;
	};
}

#endif //BASIS_HPP

#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP

#include "ElementAssemblyValues.hpp"
#include <Eigen/Dense>

namespace poly_fem
{
	class Laplacian
	{
	public:
		Eigen::Matrix<double, 1, 1> assemble(const ElementAssemblyValues &vals, const int i, const int j, const Eigen::VectorXd &da) const;

		inline int size() const { return 1; }
	};
}

#endif //LAPLACIAN_HPP

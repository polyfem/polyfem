#pragma once

#include "Common.hpp"

#include "ElementAssemblyValues.hpp"
#include <Eigen/Dense>

namespace poly_fem
{
	class Helmholtz
	{
	public:
		Eigen::Matrix<double, 1, 1> assemble(const ElementAssemblyValues &vals, const int i, const int j, const Eigen::VectorXd &da) const;

		inline int size() const { return 1; }

		void set_parameters(const json &params);

	private:
		double k_ = 1;
	};
}


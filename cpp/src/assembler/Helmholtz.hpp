#pragma once

#include "Common.hpp"
#include "AutodiffTypes.hpp"

#include "ElementAssemblyValues.hpp"
#include <Eigen/Dense>

namespace poly_fem
{
	class Helmholtz
	{
	public:
		Eigen::Matrix<double, 1, 1> assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const;
		Eigen::Matrix<double, 1, 1> compute_rhs(const AutodiffHessianPt &pt) const;

		Eigen::Matrix<AutodiffPt, Eigen::Dynamic, 1, 0, 3, 1> kernel(const int dim, const AutodiffPt &r) const;

		inline int size() const { return 1; }

		void set_parameters(const json &params);

	private:
		double k_ = 1;
	};
}


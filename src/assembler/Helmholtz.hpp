#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/AutodiffTypes.hpp>

#include <polyfem/ElementAssemblyValues.hpp>
#include <Eigen/Dense>

//local assembler for helmolhz equation, see Laplace
namespace polyfem
{
	class Helmholtz
	{
	public:
		Eigen::Matrix<double, 1, 1> assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const;
		Eigen::Matrix<double, 1, 1> compute_rhs(const AutodiffHessianPt &pt) const;

		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> kernel(const int dim, const AutodiffScalarGrad &r) const;

		inline int size() const { return 1; }

		//sets the k parameter
		void set_parameters(const json &params);

	private:
		double k_ = 1;
	};
} // namespace polyfem

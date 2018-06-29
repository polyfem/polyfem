#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP

#include <polyfem/Common.hpp>
#include <polyfem/AutodiffTypes.hpp>

#include <polyfem/ElementAssemblyValues.hpp>
#include <Eigen/Dense>

namespace polyfem
{
	class Laplacian
	{
	public:
		Eigen::Matrix<double, 1, 1> assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const;
		Eigen::Matrix<double, 1, 1> compute_rhs(const AutodiffHessianPt &pt) const;

		Eigen::Matrix<AutodiffPt, Eigen::Dynamic, 1, 0, 3, 1> kernel(const int dim, const AutodiffPt &r) const;

		inline int size() const { return 1; }

		void set_parameters(const json &params) { }
	};
}

#endif //LAPLACIAN_HPP

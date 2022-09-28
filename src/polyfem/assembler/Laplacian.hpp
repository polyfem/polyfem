#pragma once

#include "AssemblerData.hpp"

#include <polyfem/Common.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

#include <Eigen/Dense>

// local assembler for laplace equation
namespace polyfem
{
	namespace assembler
	{
		class Laplacian
		{
		public:
			// computes local stiffness matrix (1x1) for bases i,j
			Eigen::Matrix<double, 1, 1> assemble(const LinearAssemblerData &data) const;

			// uses autodiff to compute the rhs for a fabricated solution
			// in this case it just return pt.getHessian().trace()
			// pt is the evaluation of the solution at a point
			Eigen::Matrix<double, 1, 1> compute_rhs(const AutodiffHessianPt &pt) const;

			// kernel of the pde, used in kernel problem
			Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> kernel(const int dim, const AutodiffScalarGrad &r) const;

			// this is a scalar assembler, size is always 1
			inline int size() const { return 1; }

			// laplacian has no parameters.
			// in case these are passes trough params
			void add_multimaterial(const int index, const json &params) {}
		};
	} // namespace assembler
} // namespace polyfem

#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

// local assembler for laplace equation
namespace polyfem
{
	namespace assembler
	{
		class Laplacian : public ScalarLinearAssembler
		{
		public:
			using ScalarLinearAssembler::assemble;

			// computes local stiffness matrix (1x1) for bases i,j
			Eigen::Matrix<double, 1, 1> assemble(const LinearAssemblerData &data) const override;

			// uses autodiff to compute the rhs for a fabricated solution
			// in this case it just return pt.getHessian().trace()
			// pt is the evaluation of the solution at a point
			Eigen::Matrix<double, 1, 1> compute_rhs(const AutodiffHessianPt &pt) const;

			// kernel of the pde, used in kernel problem
			Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> kernel(const int dim, const AutodiffScalarGrad &r) const;

			// laplacian has no parameters.
			// in case these are passes trough params
			void add_multimaterial(const int index, const json &params) {}
		};
	} // namespace assembler
} // namespace polyfem

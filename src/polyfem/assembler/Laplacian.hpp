#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

// local assembler for laplace equation
namespace polyfem
{
	namespace assembler
	{
		class Laplacian : public LinearAssembler
		{
		public:
			using LinearAssembler::assemble;

			std::string name() const override { return "Laplacian"; }
			std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }

			// computes local stiffness matrix (1x1) for bases i,j
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> assemble(const LinearAssemblerData &data) const override;

			// uses autodiff to compute the rhs for a fabricated solution
			// in this case it just return pt.getHessian().trace()
			// pt is the evaluation of the solution at a point
			VectorNd compute_rhs(const AutodiffHessianPt &pt) const override;

			// kernel of the pde, used in kernel problem
			Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> kernel(const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const override;
		};
	} // namespace assembler
} // namespace polyfem

#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/AutodiffTypes.hpp>

#include <polyfem/ElementAssemblyValues.hpp>
#include <Eigen/Dense>

//local assembler for laplace equation
namespace polyfem
{
	namespace assembler
	{
		class Laplacian
		{
		public:
			//computes local stiffness matrix (1x1) for bases i,j
			//vals stores the evaluation for that element
			//da contains both the quadrature weight and the change of metric in the integral
			Eigen::Matrix<double, 1, 1> assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const;

			//uses autodiff to compute the rhs for a fabricated solution
			//in this case it just return pt.getHessian().trace()
			//pt is the evaluation of the solution at a point
			Eigen::Matrix<double, 1, 1> compute_rhs(const AutodiffHessianPt &pt) const;

			//kernel of the pde, used in kernel problem
			Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> kernel(const int dim, const AutodiffScalarGrad &r) const;

			//this is a scalar assembler, size is always 1
			inline int size() const { return 1; }

			//laplacian has no parameters.
			//in case these are passes trough params
			void set_parameters(const json &params) {}
		};
	} // namespace assembler
} // namespace polyfem

#pragma once

#include <polyfem/basis/Basis.hpp>

#include <Eigen/Dense>

// stores per local bases evaluations
namespace polyfem
{
	namespace assembler
	{
		class AssemblyValues
		{
		public:
			// Weighted sum to express the current ("virtual") node as a linear-combination
			// of the real (unknown) nodes
			std::vector<basis::Local2Global> global;

			// Evaluation of the basis over the quadrature points of the element
			Eigen::MatrixXd val; // R^m

			// Gradient of the basis over the quadrature points
			Eigen::MatrixXd grad; // R^{m x dim}

			// Gradient of the basis pre-multiplied by the inverse transpose of the
			// Jacobian of the geometric mapping of the element
			Eigen::MatrixXd grad_t_m; // J^{-T}*∇φi per row R^{m x dim}

			void finalize()
			{
				grad_t_m.resize(grad.rows(), grad.cols());
			}
		};
	} // namespace assembler
} // namespace polyfem

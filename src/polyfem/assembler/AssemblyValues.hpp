#pragma once

#include <polyfem/basis/Basis.hpp>

#include <Eigen/Dense>

namespace polyfem::assembler
{
	/// @brief Per local bases evaluations.
	/// @note \f$m\f$ = number of quadrature points; \f$d\f$ = dimension
	struct AssemblyValues
	{
		/// @brief Weighted sum to express the current ("virtual") node as a
		/// linear-combination of the real (unknown) nodes.
		std::vector<basis::Local2Global> global;

		/// @brief Evaluation of the basis over the quadrature points of the element (\f$\in\mathbb{R}^m\f$).
		Eigen::MatrixXd val;

		/// @brief Gradient of the basis over the quadrature points (\f$\in\mathbb{R}^{m \times d}\f$).
		Eigen::MatrixXd grad;

		/// @brief Gradient of the basis pre-multiplied by the inverse transpose
		/// of the Jacobian of the geometric mapping of the element.
		/// \f$J^{-T} \nabla \phi_i\f$ per row \f$\in\mathbb{R}^{m \times d}\f$
		Eigen::MatrixXd grad_t_m;

		void resize_grad_t_m()
		{
			grad_t_m.resize(grad.rows(), grad.cols());
		}
	};
} // namespace polyfem::assembler

#pragma once

#include <Eigen/Dense>

namespace polyfem::quadrature
{
	class Quadrature
	{
	public:
		Eigen::MatrixXd points;
		Eigen::VectorXd weights;

		static int bc_order(const int basis_degree, const int dim);

		static int stiffness_poly_order(const int basis_degree, const int dim);

		static int mass_poly_order(const int basis_degree, const int dim);

		static int stiffness_spline_order(const int basis_degree, const int dim);

		static int mass_spline_order(const int basis_degree, const int dim);

		static int stiffness_order(const int basis_degree, const int dim, const bool is_hex);

		static int mass_order(const int basis_degree, const int dim, const bool is_hex);
	};
} // namespace polyfem::quadrature

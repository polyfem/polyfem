#include "RBFWithLinear.hpp"
#include "PolygonQuadrature.hpp"
#include "Types.hpp"
#include <igl/Timer.h>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <array>


namespace poly_fem
{
	namespace
	{
		double kernel(const bool is_volume, const double r)
		{
			if(r < 1e-8) return 0;

			if(is_volume)
				return 1/r;

			return log(r);
		}

		double kernel_prime(const bool is_volume, const double r)
		{
			if(r < 1e-8) return 0;

			if(is_volume)
				return -1/(r*r);

			return 1/r;
		}
	}

	RBFWithLinear::RBFWithLinear(const Eigen::MatrixXd &centers, const Eigen::MatrixXd &samples,
		const Eigen::MatrixXd &local_basis_integral, const Quadrature &quadr, Eigen::MatrixXd &rhs)
	: centers_(centers)
	{
		compute(samples, local_basis_integral, quadr, rhs);
	}

	void RBFWithLinear::create_matrix(const Eigen::MatrixXd &samples, Eigen::MatrixXd &A) const {
		// Compute A
		const int num_kernels = centers_.rows();
		const int dim = centers_.cols();

		A.resize(samples.rows(), num_kernels + dim + 1);
		for (int j = 0; j < num_kernels; ++j) {
			A.col(j) = (samples.rowwise() - centers_.row(j)).rowwise().norm().unaryExpr([this](double x)
				{ return kernel(is_volume_, x); });
		}
		A.col(num_kernels).setOnes(); // constant term
		A.rightCols(dim) = samples; // linear terms
	}

	void RBFWithLinear::basis(const int local_index, const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
	{
		// Compute A
		Eigen::MatrixXd A;
		create_matrix(samples, A);

		// Multiply by the weights
		val = A * weights_.col(local_index);
	}

	void RBFWithLinear::grad(const int local_index, const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
	{
		// Compute Ar
		const int num_kernels = centers_.rows();
		const int dim = centers_.cols();

		std::array<Eigen::MatrixXd, 3> Axyz;
		for (int d = 0; d < dim; ++d) {
			Axyz[d].resize(samples.rows(), num_kernels + dim + 1);
			Axyz[d].setZero();
		}
		for (int j = 0; j < num_kernels; ++j) {
			Axyz[0].col(j) = (samples.rowwise() - centers_.row(j)).rowwise().norm().unaryExpr([this](double x)
				{ return kernel_prime(is_volume_, x) / x; });
			for (int d = dim; d--;) {
				Axyz[d].col(j) = (samples.col(d).array() - centers_(j, d)) * Axyz[0].col(j).array();
			}
		}
		// Linear terms
		for (int d = 0; d < dim; ++d) {
			Axyz[d].middleCols(num_kernels + 1 + d, 1).setOnes();
		}

		// Apply weights
		val.resize(samples.rows(), dim);
		for (int d = 0; d < dim; ++d) {
			val.col(d) = Axyz[d] * weights_.col(local_index);
		}
	}

	void RBFWithLinear::compute(const Eigen::MatrixXd &samples, const Eigen::MatrixXd &local_basis_integral,
		const Quadrature &quadr, Eigen::MatrixXd &rhs)
	{
		is_volume_ = samples.cols() == 3;

		std::cout << "#kernel centers: " << centers_.rows() << std::endl;
		std::cout << "#collocation points: " << samples.rows() << std::endl;
		std::cout << "#non-vanishing bases: " << rhs.cols() << std::endl;

#if 0
		// Compute A
		Eigen::MatrixXd A;
		create_matrix(samples, A);

		// Solve the system
		const int num_kernels = centers_.rows();
		std::cout << "-- Solving system of size " << num_kernels << " x " << num_kernels << std::endl;
		weights_ = (A.transpose() * A).ldlt().solve(A.transpose() * rhs);
		std::cout << "-- Solved!" << std::endl;
#else
		// For each basis function f that is nonzero on the element E, we want to
		// solve the least square system A w = rhs, where:
		//     ┏                    ┓
		//     ┃ φj(pi) ... 1 xi yi ┃
		// A = ┃   ┊        ┊  ┊  ┊ ┃ ∊ ℝ^{#S x (#K+dim+1)}
		//     ┃   ┊        ┊  ┊  ┊ ┃
		//     ┗                    ┛
		//     ┏                    ┓^⊤
		// w = ┃ wj ... a00 a10 a01 ┃   ∊ ℝ^{#K+dim+1}
		//     ┗                    ┛
		// - A is the RBF kernels evaluated over the collocation points (#S)
		// - b is the expected value of the basis sampled on the boundary (#S)
		// - w is the weight of the kernels defining the basis
		// - pi = (xi, yi) is the i-th collocation point
		//
		// Moreover, we want to impose a constraint on the gradients of the kernels
		// so that the integral of the gradients over the polytope must be equal to
		// the value specified in the argument `local_basis_integral` (#K)
		//
		// Let `lb` be the precomputed expected value of ∫f over the rest of the mesh.
		// We write down the constraint as:
		//
		// ∫_{p ∊ E} Σ_j wj ∇x(φj)(p) + ∇x(a^⊤·p + c) dp = lb       (1)
		// (1) ⇔ ∫_{p ∊ E} Σ_j wj ∇x(φj)(p) + ax dp = lb
		//     ⇔ lb - Σ_j wj ∫_{p ∊ E} ∇x(φj)(p) dp = ax Vol(E)
		//
		// We now have a relationship w = Lv + t, where the weights (and esp. the
		// linear terms in the weight vector w), are expressed as an affine
		// combination of unknowns v = [wj ... c] ∊ ℝ^{#K+1} and a translation t
		//
		// After solving the new least square system A L v = rhs - A t, we can retrieve
		// w = L v

		//
		//     ┏                      ┓^⊤
		// t = ┃ 0  ┈  ┈  0 0 lbx lby ┃   / Vol(E) ∊ ℝ^{#K+dim+1}
		//     ┗                      ┛
		//
		//     ┏                  ┓
		//     ┃   1              ┃
		//     ┃       1          ┃
		//     ┃          ·       ┃
		// L = ┃             ·    ┃ ∊ ℝ^{ (#K+dim+1) x (#K+1}) }
		//     ┃                1 ┃
		//     ┃ Lx_j  ┈        0 ┃
		//     ┃ Ly_j  ┈        0 ┃
		//     ┗                  ┛
		// Where Lx_j = -∫∇xφj / Vol(E) = -∫_{p ∊ E} ∇x(φj)(p) / Vol(E) is integrated numerically
		//

		const int num_bases = rhs.cols();
		const int num_kernels = centers_.rows();
		const int dim = centers_.cols();

		// Compute KI
		Eigen::MatrixXd KI(num_kernels, dim);
		for (int j = 0; j < num_kernels; ++j) {
			// ∫∇x(φj)(p) = Σ_q (xq - xk) * 1/r * h'(r) * wq
			// - xq is the x coordinate of the q-th quadrature point
			// - wq is the q-th quadrature weight
			// - r is the distance from pq to the kernel center
			// - h is the RBFWithLinear RBF kernel (scalar function)
			const Eigen::MatrixXd drdp = quadr.points.rowwise() - centers_.row(j);
			const Eigen::VectorXd r = drdp.rowwise().norm();
			KI.row(j) = (drdp.array().colwise() * (quadr.weights.array() * r.unaryExpr([this](double x)
				{ return kernel_prime(is_volume_, x); }).array() / r.array())).colwise().sum();
		}
		KI /= quadr.weights.sum();

		// Compute A
		Eigen::MatrixXd A;
		create_matrix(samples, A);

		// Compute L
		Eigen::MatrixXd L(num_kernels + dim + 1, num_kernels + 1);
		L.setZero();
		L.diagonal().setOnes();
		L.block(num_kernels + 1, 0, dim, num_kernels) = -KI.transpose();
		std::cout << L.bottomRightCorner(10, 10) << std::endl;

		// Compute t
		weights_.resize(A.cols(), num_bases);
		weights_.setZero();
		weights_.bottomRows(dim) = local_basis_integral.transpose() / quadr.weights.sum();

		// Compute b = rhs - A t
		rhs -= A * weights_;

		// Solve the system
		std::cout << "-- Solving system of size " << num_kernels << " x " << num_kernels << std::endl;
		weights_ += L * (L.transpose() * A.transpose() * A * L).ldlt().solve(L.transpose() * A.transpose() * rhs);
		std::cout << "-- Solved!" << std::endl;

		// std::cout << weights_.bottomRows(10) << std::endl;

		// Eigen::MatrixXd M, x, dx;
		// grad(0, quadr.points, M);
		// for (int d = 0; d < dim; ++d) {
		// 	basis(0, quadr.points, x);
		// 	auto asd = quadr.points;
		// 	asd.col(d).array() += 1e-7;
		// 	basis(0, asd, dx);
		// 	std::cout << (dx - x) / 1e-7 - M.col(d) << std::endl;
		// 	std::cout << (M.col(d).array() * quadr.weights.array()).sum() - local_basis_integral(0, d) << std::endl;
		// }
#endif
	}
}

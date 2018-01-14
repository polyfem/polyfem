#include "RBFWithQuadratic.hpp"
#include "Types.hpp"
#include <igl/Timer.h>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <array>

int num_removed = 0;


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

		// double kernel(const bool is_volume, const double r)
		// {
		// 	if(r < 1e-8) return 0;

		// 	return r * r * (log(r)-1);
		// }

		// double kernel_prime(const bool is_volume, const double r)
		// {
		// 	if(r < 1e-8) return 0;

		// 	return r * ( 2 * log(r) - 1);
		// }
	}

	RBFWithQuadratic::RBFWithQuadratic(const Eigen::MatrixXd &centers, const Eigen::MatrixXd &samples,
		const Eigen::MatrixXd &local_basis_integral, const Quadrature &quadr, Eigen::MatrixXd &rhs)
	: centers_(centers)
	{
		centers_.resize(0, 2);
		compute(samples, local_basis_integral, quadr, rhs);
	}

	void RBFWithQuadratic::create_matrix(const Eigen::MatrixXd &samples, Eigen::MatrixXd &A) const {
		// Compute A
		const int num_kernels = centers_.rows();
		const int dim = centers_.cols();

		A.resize(samples.rows(), num_kernels + dim*(dim+1)/2 + dim + 1);
		for (int j = 0; j < num_kernels; ++j) {
			A.col(j) = (samples.rowwise() - centers_.row(j)).rowwise().norm().unaryExpr([this](double x)
				{ return kernel(is_volume_, x); });
		}
		A.col(num_kernels).setOnes(); // constant term
		A.middleCols(num_kernels + 1, dim) = samples; // linear terms
		if (dim == 2) {
			A.middleCols(num_kernels + dim + 1, 1) = samples.rowwise().prod(); // mixed terms
		} else if (dim == 3) {
			A.middleCols(num_kernels + dim + 1, 3) = samples;
			A.middleCols(num_kernels + dim + 1 + 0, 1) *= samples.col(1);
			A.middleCols(num_kernels + dim + 1 + 1, 1) *= samples.col(2);
			A.middleCols(num_kernels + dim + 1 + 2, 1) *= samples.col(0);
		}
		A.rightCols(dim) = samples.array().square(); // quadratic terms

		A.conservativeResize(A.rows(), A.cols() - num_removed);
	}

	// void RBFWithQuadratic::create_matrix(const Eigen::MatrixXd &samples, Eigen::MatrixXd &A) const {
	// 	// Compute A
	// 	const int num_kernels = centers_.rows();
	// 	const int dim = centers_.cols();
	// 	// const int num_cols = num_kernels + dim + 1;
	// 	const int num_cols = num_kernels + dim*(dim+1)/2 + dim + 1;
	// 	const int num_squared = dim;

	// 	A.resize(samples.rows(), num_cols);
	// 	for (int j = 0; j < num_kernels; ++j) {
	// 		A.col(j) = (samples.rowwise() - centers_.row(j)).rowwise().norm().unaryExpr([this](double x)
	// 			{ return kernel(is_volume_, x); });
	// 	}
	// 	A.middleCols(num_kernels, dim) = samples.array().square(); // quadratic terms
	// 	if (dim == 2) {
	// 		A.middleCols(num_kernels + num_squared, 1) = samples.rowwise().prod(); // mixed terms
	// 	} else if (dim == 3) {
	// 		A.middleCols(num_kernels + num_squared, 3) = samples;
	// 		A.middleCols(num_kernels + num_squared + 0, 1) *= samples.col(1);
	// 		A.middleCols(num_kernels + num_squared + 1, 1) *= samples.col(2);
	// 		A.middleCols(num_kernels + num_squared + 2, 1) *= samples.col(0);
	// 	}
	// 	A.middleCols(A.cols() - dim - 1, dim) = samples; // linear terms
	// 	A.rightCols<1>().setOnes();
	// }

	void RBFWithQuadratic::basis(const int local_index, const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
	{
		// Compute A
		Eigen::MatrixXd A;
		create_matrix(samples, A);

		// Multiply by the weights
		val = A * weights_.col(local_index);
	}

	// void RBFWithQuadratic::grad(const int local_index, const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
	// {
	// 	// Compute Ar
	// 	const int num_kernels = centers_.rows();
	// 	const int dim = centers_.cols();
	// 	// const int num_cols = num_kernels + dim + 1;
	// 	const int num_cols = num_kernels + dim*(dim+1)/2 + dim + 1;
	// 	const int num_squared = dim;

	// 	std::array<Eigen::MatrixXd, 3> Axyz;
	// 	for (int d = 0; d < dim; ++d) {
	// 		Axyz[d].resize(samples.rows(), num_cols);
	// 		Axyz[d].setZero();
	// 	}
	// 	for (int j = 0; j < num_kernels; ++j) {
	// 		Axyz[0].col(j) = (samples.rowwise() - centers_.row(j)).rowwise().norm().unaryExpr([this](double x)
	// 			{ return kernel_prime(is_volume_, x) / x; });
	// 		for (int d = dim; d--;) {
	// 			Axyz[d].col(j) = (samples.col(d).array() - centers_(j, d)) * Axyz[0].col(j).array();
	// 		}
	// 	}
	// 	// Quadratic terms
	// 	for (int d = 0; d < dim; ++d) {
	// 		Axyz[d].middleCols(num_kernels + d, 1) = 2.0 * samples.col(d);
	// 	}
	// 	// Mixed terms
	// 	for (int d = 0; d < dim; ++d) {
	// 		if (dim == 2) {
	// 			Axyz[d].middleCols(num_kernels + num_squared, 1) = samples.col(1 - d);
	// 		} else {
	// 			assert(false);
	// 		}
	// 	}
	// 	// Linear terms
	// 	for (int d = 0; d < dim; ++d) {
	// 		Axyz[d].middleCols(num_cols - dim - 1 + d, 1).setOnes();
	// 	}

	// 	// Apply weights
	// 	val.resize(samples.rows(), dim);
	// 	for (int d = 0; d < dim; ++d) {
	// 		val.col(d) = Axyz[d] * weights_.col(local_index);
	// 	}
	// }

	void RBFWithQuadratic::grad(const int local_index, const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
	{
		// Compute Ar
		const int num_kernels = centers_.rows();
		const int dim = centers_.cols();

		std::array<Eigen::MatrixXd, 3> Axyz;
		for (int d = 0; d < dim; ++d) {
			Axyz[d].resize(samples.rows(), num_kernels + dim*(dim+1)/2 + dim + 1);
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
			Axyz[d].col(num_kernels + 1 + d).setOnes();
		}
		// Mixed terms
		for (int d = 0; d < dim; ++d) {
			if (dim == 2) {
				Axyz[d].col(num_kernels + 1 + dim) = samples.col(1 - d);
			} else {
				assert(false);
			}
		}
		// Quadratic terms
		for (int d = 0; d < dim; ++d) {
			Axyz[d].col(Axyz[d].cols() - dim + d) = 2.0 * samples.col(d);
			Axyz[d].conservativeResize(Axyz[d].rows(), Axyz[d].cols() - num_removed);
		}

		// std::cout << Axyz[0].topRightCorner(10, 10) << std::endl << std::endl;
		// std::cout << Axyz[1].topRightCorner(10, 10) << std::endl << std::endl;

		// Apply weights
		val.resize(samples.rows(), dim);
		for (int d = 0; d < dim; ++d) {
			val.col(d) = Axyz[d] * weights_.col(local_index);
		}
	}

	void RBFWithQuadratic::compute(const Eigen::MatrixXd &samples, const Eigen::MatrixXd &local_basis_integral,
		const Quadrature &quadr, Eigen::MatrixXd &rhs)
	{
		is_volume_ = (samples.cols() == 3);

		std::cout << "#kernel centers: " << centers_.rows() << std::endl;
		std::cout << "#collocation points: " << samples.rows() << std::endl;
		std::cout << "#quadrature points: " << quadr.weights.size() << std::endl;
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
		// For each shape function N that is nonzero on the element E, we want to
		// solve the least square system A w = rhs, where:
		//     ┏                                    ┓
		//     ┃ φj(pi) ... 1 xi yi xi*yi xi^2 yi^2 ┃
		// A = ┃   ┊        ┊  ┊  ┊   ┊    ┊    ┊   ┃ ∊ ℝ^{#S x (#K+1+dim+dim*(dim+1)/2)}
		//     ┃   ┊        ┊  ┊  ┊   ┊    ┊    ┊   ┃
		//     ┗                                    ┛
		//     ┏                                ┓^⊤
		// w = ┃ wj ... a00 a10 a01 a11 a20 a02 ┃   ∊ ℝ^{#K+1+dim+dim*(dim+1)/2}
		//     ┗                                ┛
		// - A is the RBF kernels evaluated over the collocation points (#S)
		// - b is the expected value of the basis sampled on the boundary (#S)
		// - w is the weight of the kernels defining the basis
		// - pi = (xi, yi) is the i-th collocation point
		//
		// Moreover, we want to impose a constraint on the weight vectors w so that
		// each monomial x^α*y^β with α+β <= 2 satisfies our PDE. If we want to
		// integrate each monomial exactly for the Laplacian PDE, then each each
		// x^α*y^β must satisfy the following:
		//
		//     Δ(x^α*y^β) = α*(α-1)*x^(α-2)*y^β + β*(β-1)*x^α*y^(β-2) = hαβ(x,y)
		//
		// In the case of (α,β) ∊ {(1,0), (0,1), (1,1), (2,0), (0,2)}, this simplifies
		// to the five equations:
		//
		//     Δx  = 0      (1a)
		//     Δy  = 0      (1b)
		//     Δxy = 0      (1c)
		//     Δx² = 1      (1d)
		//     Δy² = 1      (1e)
		//
		// If our bases {N_k}_k integrated exactly those monomials, then the weak form
		// of the PDE must be satisfied, for each basis/shape function N_k :
		//
		//     ∫_Ω ∇(x^α*y^β)·∇(N_k) = hαβ(x,y)
		//
		// Now, if we consider our polytope element E, and split the integral above
		// between E and Ω\E, we arrive at the constraint that
		//
		//     ∫_E ∇(x^α*y^β)·∇(N_k) + cαβ = hαβ(x,y)
		//
		// Where cαβ is a constant term corresponding to the integral over the
		// other elements of the mesh. Now, the five equations in (1) lead to:
		//
		//     ∫_E ∇x(N_k) + c10 = 0                       (2a)
		//     ∫_E ∇y(N_k) + c01 = 0                       (2b)
		//     ∫_E (y·∇x(N_k) + y·∇x(N_k)) + c11 = 0       (2c)
		//     ∫_E 2x·∇x(N_k) + c20 = 2                    (2d)
		//     ∫_E 2y·∇y(N_k) + c02 = 2                    (2e)
		//
		// After writing the shape function N_k as:
		//
		//     N_k(x,y) = Σ_j wj φj(x,y) + a00 + a10*x + a01*y + a11*x*y + a20*x² + a02*y²
		//
		// The five equations (2) become:
		//
		//    Σ_j wj ∫∇x(φj) + a10 |E| + a11 ∫y + a20 ∫2x + c10 = 0
		//    Σ_j wj ∫∇y(φj) + a01 |E| + a11 ∫x + a02 ∫2y + c01 = 0
		//    Σ_j wj (∫y·∇x(φj) + ∫x·∇y(φj)) + a10 ∫y + a01 ∫x + a11 (∫x²+∫y²) + a20 2∫xy + a02 2∫xy + c11 = 0
		//    Σ_j wj 2∫x·∇x(φj) + a10 2∫x + a11 2∫xy + a20 4∫x² + c20 = 2
		//    Σ_j wj 2∫y·∇y(φj) + a01 2∫y + a11 2∫xy + a02 4∫y² + c02 = 2
		//
		// This system gives us a relationship between the fives a10, a01, a11, a20, a02
		// and the rest of the wj + a constant translation term. We can write down the
		// following relationship:
		//
		//       a10   a01   a11   a20   a02
		//     ┏                              ┓             ┏    ┓
		//     ┃ |E|         ∫y    2∫x        ┃             ┃ wj ┃
		//     ┃                              ┃             ┃ ┊  ┃
		//     ┃       |E|   ∫x          2∫y  ┃             ┃ ┊  ┃
		//     ┃                              ┃             ┃ ┊  ┃
		// M = ┃  ∫y   ∫x  ∫x²+∫y² 2∫xy  2∫xy ┃ = \tilde{L} ┃ ┊  ┃ + \tilde{t}
		//     ┃                              ┃             ┃ ┊  ┃
		//     ┃ 2∫x       2∫xy    4∫x²       ┃             ┃ ┊  ┃
		//     ┃                              ┃             ┃w_#K┃
		//     ┃      2∫y  2∫xy          4∫y² ┃             ┃ a00┃
		//     ┗                              ┛             ┗    ┛
		//
		// Now, if we want to express w as w = Lv + t, and solve our least-square
		// system as before, we need to invert M and compute L and t in terms of
		// \tilde{L} and \tilde{t}
		//
		//     ┏                  ┓
		//     ┃   1              ┃
		//     ┃       1          ┃
		//     ┃          ·       ┃
		// L = ┃             ·    ┃ ∊ ℝ^{ (#K+1+dim+dim*(dim+1)/2) x (#K+1}) }
		//     ┃                1 ┃
		//     ┃ M^{-1} \tilde{L} ┃
		//     ┗                  ┛
		//     ┏                  ┓
		//     ┃        0         ┃
		//     ┃        ┊         ┃
		// t = ┃        ┊         ┃ ∊ ℝ^{#K+1+dim+dim*(dim+1)/2}
		//     ┃        0         ┃
		//     ┃ M^{-1} \tilde{t} ┃
		//     ┗                  ┛
		// After solving the new least square system A L v = rhs - A t, we can retrieve
		// w = L v

		const int num_bases = rhs.cols();
		const int num_kernels = centers_.rows();
		const int dim = centers_.cols();

		// K_lin = ∫∇x(φj), ∫∇y(φj)
		// K_mix = ∫y·∇x(φj), ∫x·∇y(φj)
		// K_sqr = ∫x·∇x(φj), ∫y·∇y(φj)
		Eigen::MatrixXd K_lin = Eigen::MatrixXd::Zero(num_kernels, dim);
		Eigen::MatrixXd K_mix = Eigen::MatrixXd::Zero(num_kernels, dim);
		Eigen::MatrixXd K_sqr = Eigen::MatrixXd::Zero(num_kernels, dim);
		for (int j = 0; j < num_kernels; ++j) {
			// ∫∇x(φj)(p) = Σ_q (xq - xk) * 1/r * h'(r) * wq
			// - xq is the x coordinate of the q-th quadrature point
			// - wq is the q-th quadrature weight
			// - r is the distance from pq to the kernel center
			// - h is the RBF kernel (scalar function)
			for (int q = 0; q < quadr.points.rows(); ++q) {
				const RowVectorNd p = quadr.points.row(q) - centers_.row(j);
				const double r = p.norm();
				const RowVectorNd gradPhi = p * kernel_prime(is_volume_, r) / r * quadr.weights(q);
				K_lin.row(j) += gradPhi;
				K_mix(j,0) += quadr.points(q,1)*gradPhi(0);
				K_mix(j,1) += quadr.points(q,0)*gradPhi(1);
				K_sqr.row(j) += (quadr.points.row(q).array() * gradPhi.array()).matrix();
			}
		}

		// I_lin = ∫x, ∫y
		// I_mix = ∫xy
		// I_sqr = ∫x², ∫y²
		Eigen::RowVectorXd I_lin = (quadr.points.array().colwise() * quadr.weights.array()).colwise().sum();
		Eigen::RowVectorXd I_mix = (quadr.points.rowwise().prod().array() * quadr.weights.array()).colwise().sum();
		Eigen::RowVectorXd I_sqr = (quadr.points.array().square().colwise() * quadr.weights.array()).colwise().sum();
		double volume = quadr.weights.sum();

		// std::cout << I_lin << std::endl;
		// std::cout << I_mix << std::endl;
		// std::cout << I_sqr << std::endl;

		// Compute M
		Eigen::Matrix<double, 5, 5> M;
		M <<    volume,          0,          I_lin(1), 2*I_lin(0),          0,
			         0,     volume,          I_lin(0),          0, 2*I_lin(1),
			  I_lin(1),   I_lin(0), I_sqr(0)+I_sqr(1), 2*I_mix(0), 2*I_mix(0),
			4*I_lin(0), 2*I_lin(1),        4*I_mix(0), 6*I_sqr(0), 2*I_sqr(1),
			2*I_lin(0), 4*I_lin(1),        4*I_mix(0), 2*I_sqr(0), 6*I_sqr(1);
		Eigen::Matrix<double, 5, 5> Minv = M.inverse();

		// std::cout << M << std::endl;
		// std::cout << M.determinant() << std::endl;

		// Compute A
		Eigen::MatrixXd A;
		create_matrix(samples, A);

		// Compute L
		Eigen::MatrixXd L(num_kernels + 1 + dim + dim*(dim+1)/2, num_kernels + 1);
		L.setZero();
		L.diagonal().setOnes();
		L.block(num_kernels + 1, 0, dim, num_kernels) = -K_lin.transpose();
		L.block(num_kernels + 1 + dim, 0, 1, num_kernels) = -K_mix.transpose().colwise().sum();
		L.block(num_kernels + 1 + dim + 1, 0, dim, num_kernels) = -2.0 * K_sqr.transpose();
		L.bottomRightCorner(dim, 1).setConstant(-2.0 * volume);
		L.block(num_kernels + 1, 0, 5, num_kernels + 1) = (Minv * L.block(num_kernels + 1, 0, 5, num_kernels + 1)).eval();
		L.conservativeResize(L.rows() - num_removed, L.cols());
		// std::cout << L << std::endl;
		// std::cout << L.bottomRightCorner(5, 5) << std::endl;

		// Compute t
		weights_.resize(A.cols(), num_bases);
		weights_.setZero();
		weights_.bottomRows(5 - num_removed) = local_basis_integral.transpose();
		weights_.bottomRows(5 - num_removed) = (Minv * weights_.bottomRows(5 - num_removed)).eval();

		// Compute b = rhs - A t
		rhs -= A * weights_;

		// Solve the system
		std::cout << "-- Solving system of size " << L.cols() << " x " << L.cols() << std::endl;
		weights_ += L * (L.transpose() * A.transpose() * A * L).ldlt().solve(L.transpose() * A.transpose() * rhs);
		std::cout << "-- Solved!" << std::endl;

		// std::cout << weights_.leftCols(10) << std::endl << std::endl;
		// std::cout << local_basis_integral.row(0).transpose() << std::endl << std::endl;

		// std::cout << M * weights_.bottomLeftCorner(5, 1) - local_basis_integral.row(0).transpose() << std::endl << std::endl;

#endif
		// Eigen::MatrixXd MM, x, dx;
		// grad(0, quadr.points, MM);
		// for (int d = 0; d < 2; ++d) {
		// 	basis(0, quadr.points, x);
		// 	auto asd = quadr.points;
		// 	asd.col(d).array() += 1e-7;
		// 	basis(0, asd, dx);
		// 	// std::cout << (dx - x) / 1e-7 - MM.col(d) << std::endl;
		// 	std::cout << (MM.col(d).array() * quadr.weights.array()).sum() - local_basis_integral(0, d) << std::endl;
		// 	std::cout << (2.0 * quadr.points.col(d).array() * MM.col(d).array() * quadr.weights.array()).sum() - - local_basis_integral(0, d+3) << std::endl;
		// }
		// std::cout << ((
		// 		MM.col(1).array() * quadr.points.col(0).array()
		// 		+ MM.col(0).array() * quadr.points.col(1).array()
		// 	) * quadr.weights.array()).sum() - local_basis_integral(0, 2) << std::endl;
	}
}

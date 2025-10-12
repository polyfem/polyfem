////////////////////////////////////////////////////////////////////////////////
#include "RBFWithLinear.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Logger.hpp>

#include <igl/Timer.h>

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <array>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::basis;
using namespace polyfem::quadrature;

namespace
{

	// Harmonic kernel
	double kernel(const bool is_volume, const double r)
	{
		if (r < 1e-8)
		{
			return 0;
		}

		if (is_volume)
		{
			return 1 / r;
		}
		else
		{
			return log(r);
		}
	}

	double kernel_prime(const bool is_volume, const double r)
	{
		if (r < 1e-8)
		{
			return 0;
		}

		if (is_volume)
		{
			return -1 / (r * r);
		}
		else
		{
			return 1 / r;
		}
	}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

RBFWithLinear::RBFWithLinear(
	const Eigen::MatrixXd &centers,
	const Eigen::MatrixXd &samples,
	const Eigen::MatrixXd &local_basis_integral,
	const Quadrature &quadr,
	Eigen::MatrixXd &rhs,
	bool with_constraints)
	: centers_(centers)
{
	compute_weights(samples, local_basis_integral, quadr, rhs, with_constraints);
}

// -----------------------------------------------------------------------------

void RBFWithLinear::basis(const int local_index, const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
{
	Eigen::MatrixXd tmp;
	bases_values(samples, tmp);
	val = tmp.col(local_index);
}

// -----------------------------------------------------------------------------

void RBFWithLinear::grad(const int local_index, const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
{
	Eigen::MatrixXd tmp;
	const int dim = centers_.cols();
	val.resize(samples.rows(), dim);
	for (int d = 0; d < dim; ++d)
	{
		bases_grads(d, samples, tmp);
		val.col(d) = tmp.col(local_index);
	}
}

////////////////////////////////////////////////////////////////////////////////

void RBFWithLinear::bases_values(const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
{
	// Compute A
	Eigen::MatrixXd A;
	compute_kernels_matrix(samples, A);

	// Multiply by the weights
	val = A * weights_;
}

// -----------------------------------------------------------------------------

void RBFWithLinear::bases_grads(const int axis, const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
{
	const int num_kernels = centers_.rows();
	const int dim = centers_.cols();

	// Compute ∇xA
	Eigen::MatrixXd A_prime(samples.rows(), num_kernels + 1 + dim);
	A_prime.setZero();

	for (int j = 0; j < num_kernels; ++j)
	{
		A_prime.col(j) = (samples.rowwise() - centers_.row(j)).rowwise().norm().unaryExpr([this](double x) { return kernel_prime(is_volume(), x) / x; });
		A_prime.col(j) = (samples.col(axis).array() - centers_(j, axis)) * A_prime.col(j).array();
	}
	// Linear terms
	A_prime.middleCols(num_kernels + 1 + axis, 1).setOnes();

	// Apply weights
	val = A_prime * weights_;
}

////////////////////////////////////////////////////////////////////////////////

void RBFWithLinear::compute_kernels_matrix(const Eigen::MatrixXd &samples, Eigen::MatrixXd &A) const
{
	// Compute A
	const int num_kernels = centers_.rows();
	const int dim = centers_.cols();

	A.resize(samples.rows(), num_kernels + 1 + dim);
	for (int j = 0; j < num_kernels; ++j)
	{
		A.col(j) = (samples.rowwise() - centers_.row(j)).rowwise().norm().unaryExpr([this](double x) { return kernel(is_volume(), x); });
	}
	A.col(num_kernels).setOnes(); // constant term
	A.rightCols(dim) = samples;   // linear terms
}

// -----------------------------------------------------------------------------

void RBFWithLinear::compute_constraints_matrix(
	const int num_bases,
	const Quadrature &quadr,
	const Eigen::MatrixXd &local_basis_integral,
	Eigen::MatrixXd &L,
	Eigen::MatrixXd &t) const
{
	const int num_kernels = centers_.rows();
	const int dim = centers_.cols();

	// Compute KI
	Eigen::MatrixXd KI(num_kernels, dim);
	for (int j = 0; j < num_kernels; ++j)
	{
		// ∫∇x(φj)(p) = Σ_q (xq - xk) * 1/r * h'(r) * wq
		// - xq is the x coordinate of the q-th quadrature point
		// - wq is the q-th quadrature weight
		// - r is the distance from pq to the kernel center
		// - h is the RBFWithLinear RBF kernel (scalar function)
		const Eigen::MatrixXd drdp = quadr.points.rowwise() - centers_.row(j);
		const Eigen::VectorXd r = drdp.rowwise().norm();
		KI.row(j) = (drdp.array().colwise() * (quadr.weights.array() * r.unaryExpr([this](double x) { return kernel_prime(is_volume(), x); }).array() / r.array())).colwise().sum();
	}
	KI /= quadr.weights.sum();

	// Compute L
	L.resize(num_kernels + dim + 1, num_kernels + 1);
	L.setZero();
	L.diagonal().setOnes();
	L.block(num_kernels + 1, 0, dim, num_kernels) = -KI.transpose();

	// Compute t
	t.resize(num_kernels + 1 + dim, num_bases);
	t.setZero();
	t.bottomRows(dim) = local_basis_integral.transpose().topRows(dim) / quadr.weights.sum();
}

// -----------------------------------------------------------------------------

void RBFWithLinear::compute_weights(const Eigen::MatrixXd &samples,
									const Eigen::MatrixXd &local_basis_integral, const Quadrature &quadr,
									Eigen::MatrixXd &rhs, bool with_constraints)
{
	logger().trace("#kernel centers: {}", centers_.rows());
	logger().trace("#collocation points: {}", samples.rows());
	logger().trace("#quadrature points: {}", quadr.weights.size());
	logger().trace("#non-vanishing bases: {}", rhs.cols());

	if (!with_constraints)
	{
		// Compute A
		Eigen::MatrixXd A;
		compute_kernels_matrix(samples, A);

		// Solve the system
		const int num_kernels = centers_.rows();
		logger().trace("-- Solving system of size {}x{}", num_kernels, num_kernels);
		weights_ = (A.transpose() * A).ldlt().solve(A.transpose() * rhs);
		logger().trace("-- Solved!");

		return;
	}

	// For each basis function f that is nonzero on the element E, we want to
	// solve the least square system A w = rhs, where:
	//     ┏                    ┓
	//     ┃ φj(pi) ... 1 xi yi ┃
	// A = ┃   ┊        ┊  ┊  ┊ ┃ ∊ ℝ^{#S x (#K+1+dim)}
	//     ┃   ┊        ┊  ┊  ┊ ┃
	//     ┗                    ┛
	//     ┏                    ┓^⊤
	// w = ┃ wj ... a00 a10 a01 ┃   ∊ ℝ^{#K+1+dim}
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
	// t = ┃ 0  ┈  ┈  0 0 lbx lby ┃   / Vol(E) ∊ ℝ^{#K+1+dim}
	//     ┗                      ┛
	//
	//     ┏                  ┓
	//     ┃   1              ┃
	//     ┃       1          ┃
	//     ┃          ·       ┃
	// L = ┃             ·    ┃ ∊ ℝ^{ (#K+1+dim) x (#K+1}) }
	//     ┃                1 ┃
	//     ┃ Lx_j  ┈        0 ┃
	//     ┃ Ly_j  ┈        0 ┃
	//     ┗                  ┛
	// Where Lx_j = -∫∇xφj / Vol(E) = -∫_{p ∊ E} ∇x(φj)(p) / Vol(E) is integrated numerically
	//

	const int num_bases = rhs.cols();

	// Compute A
	Eigen::MatrixXd A;
	compute_kernels_matrix(samples, A);

	// Compute L and t
	// Note that t is stored into `weights_` for memory efficiency reasons
	Eigen::MatrixXd L;
	compute_constraints_matrix(num_bases, quadr, local_basis_integral, L, weights_);

	// Compute b = rhs - A t
	rhs -= A * weights_;

	// Solve the system
	logger().trace("-- Solving system of size {}x{}", L.cols(), L.cols());
	weights_ += L * (L.transpose() * A.transpose() * A * L).ldlt().solve(L.transpose() * A.transpose() * rhs);
	logger().trace("-- Solved!");
}

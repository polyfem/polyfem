////////////////////////////////////////////////////////////////////////////////
#include "RBFWithQuadraticLagrange.hpp"
#include "RBFWithQuadratic.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <igl/Timer.h>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <array>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::assembler;
using namespace polyfem::basis;
using namespace polyfem::quadrature;
using namespace polyfem::utils;

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

	// Biharmonic kernel (2d only)
	// double kernel(const bool is_volume, const double r) {
	// 	assert(!is_volume);
	// 	if (r < 1e-8) { return 0; }

	// 	return r * r * (log(r)-1);
	// }

	// double kernel_prime(const bool is_volume, const double r) {
	// 	assert(!is_volume);
	// 	if (r < 1e-8) { return 0; }

	// 	return r * ( 2 * log(r) - 1);
	// }

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

RBFWithQuadraticLagrange::RBFWithQuadraticLagrange(
	const LinearAssembler &assembler,
	const Eigen::MatrixXd &centers,
	const Eigen::MatrixXd &collocation_points,
	const Eigen::MatrixXd &local_basis_integral,
	const Quadrature &quadr,
	Eigen::MatrixXd &rhs,
	bool with_constraints)
	: centers_(centers)
{
	// centers_.resize(0, centers.cols());
	compute_weights(assembler, collocation_points, local_basis_integral, quadr, rhs, with_constraints);
}

// -----------------------------------------------------------------------------

void RBFWithQuadraticLagrange::basis(const int local_index, const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
{
	Eigen::MatrixXd tmp;
	bases_values(samples, tmp);
	val = tmp.col(local_index);
}

// -----------------------------------------------------------------------------

void RBFWithQuadraticLagrange::grad(const int local_index, const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
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

void RBFWithQuadraticLagrange::bases_values(const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
{
	// Compute A
	Eigen::MatrixXd A;
	compute_kernels_matrix(samples, A);

	// Multiply by the weights
	val = A * weights_;
}

// -----------------------------------------------------------------------------

void RBFWithQuadraticLagrange::bases_grads(const int axis, const Eigen::MatrixXd &samples, Eigen::MatrixXd &val) const
{
	const int num_kernels = centers_.rows();
	const int dim = (is_volume() ? 3 : 2);

	// Compute ∇xA
	Eigen::MatrixXd A_prime(samples.rows(), num_kernels + 1 + dim + dim * (dim + 1) / 2);
	A_prime.setZero();

	for (int j = 0; j < num_kernels; ++j)
	{
		A_prime.col(j) = (samples.rowwise() - centers_.row(j)).rowwise().norm().unaryExpr([this](double x) { return kernel_prime(is_volume(), x) / x; });
		A_prime.col(j) = (samples.col(axis).array() - centers_(j, axis)) * A_prime.col(j).array();
	}
	// Linear terms
	A_prime.middleCols(num_kernels + 1 + axis, 1).setOnes();
	// Mixed terms
	if (dim == 2)
	{
		A_prime.col(num_kernels + 1 + dim) = samples.col(1 - axis);
	}
	else
	{
		A_prime.col(num_kernels + 1 + dim + axis) = samples.col((axis + 1) % dim);
		A_prime.col(num_kernels + 1 + dim + (axis + 2) % dim) = samples.col((axis + 2) % dim);
	}
	// Quadratic terms
	A_prime.rightCols(dim).col(axis) = 2.0 * samples.col(axis);

	// Apply weights
	val = A_prime * weights_;
}

////////////////////////////////////////////////////////////////////////////////

void RBFWithQuadraticLagrange::compute_kernels_matrix(const Eigen::MatrixXd &samples, Eigen::MatrixXd &A) const
{
	// Compute A
	const int num_kernels = centers_.rows();
	const int dim = (is_volume() ? 3 : 2);

	A.resize(samples.rows(), num_kernels + 1 + dim + dim * (dim + 1) / 2);
	for (int j = 0; j < num_kernels; ++j)
	{
		A.col(j) = (samples.rowwise() - centers_.row(j)).rowwise().norm().unaryExpr([this](double x) { return kernel(is_volume(), x); });
	}
	A.col(num_kernels).setOnes();                 // constant term
	A.middleCols(num_kernels + 1, dim) = samples; // linear terms
	if (dim == 2)
	{
		A.middleCols(num_kernels + dim + 1, 1) = samples.rowwise().prod(); // mixed terms
	}
	else if (dim == 3)
	{
		A.middleCols(num_kernels + dim + 1, 3) = samples;
		A.middleCols(num_kernels + dim + 1 + 0, 1).array() *= samples.col(1).array();
		A.middleCols(num_kernels + dim + 1 + 1, 1).array() *= samples.col(2).array();
		A.middleCols(num_kernels + dim + 1 + 2, 1).array() *= samples.col(0).array();
	}
	A.rightCols(dim) = samples.array().square(); // quadratic terms
}

// -----------------------------------------------------------------------------

void RBFWithQuadraticLagrange::compute_constraints_matrix_2d_old(
	const int num_bases, const Quadrature &quadr, Eigen::MatrixXd &C) const
{
	const int num_kernels = centers_.rows();
	const int dim = centers_.cols();
	assert(dim == 2);

	// K_cst = ∫φj
	// K_lin = ∫∇x(φj), ∫∇y(φj)
	// K_mix = ∫y·∇x(φj), ∫x·∇y(φj)
	// K_sqr = ∫x·∇x(φj), ∫y·∇y(φj)
	Eigen::VectorXd K_cst = Eigen::VectorXd::Zero(num_kernels);
	Eigen::MatrixXd K_lin = Eigen::MatrixXd::Zero(num_kernels, dim);
	Eigen::MatrixXd K_mix = Eigen::MatrixXd::Zero(num_kernels, dim);
	Eigen::MatrixXd K_sqr = Eigen::MatrixXd::Zero(num_kernels, dim);
	for (int j = 0; j < num_kernels; ++j)
	{
		// ∫∇x(φj)(p) = Σ_q (xq - xk) * 1/r * h'(r) * wq
		// - xq is the x coordinate of the q-th quadrature point
		// - wq is the q-th quadrature weight
		// - r is the distance from pq to the kernel center
		// - h is the RBF kernel (scalar function)
		for (int q = 0; q < quadr.points.rows(); ++q)
		{
			const RowVectorNd p = quadr.points.row(q) - centers_.row(j);
			const double r = p.norm();
			const RowVectorNd gradPhi = p * kernel_prime(is_volume(), r) / r * quadr.weights(q);
			K_cst(j) += kernel(is_volume(), r) * quadr.weights(q);
			K_lin.row(j) += gradPhi;
			K_mix(j, 0) += quadr.points(q, 1) * gradPhi(0);
			K_mix(j, 1) += quadr.points(q, 0) * gradPhi(1);
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

	// TODO
	//  std::cout << I_lin << std::endl;
	//  std::cout << I_mix << std::endl;
	//  std::cout << I_sqr << std::endl;

	// Compute M
	Eigen::Matrix<double, 5, 5> M;
	M << volume, 0, I_lin(1), 2 * I_lin(0), 0,
		0, volume, I_lin(0), 0, 2 * I_lin(1),
		I_lin(1), I_lin(0), I_sqr(0) + I_sqr(1), 2 * I_mix(0), 2 * I_mix(0),
		4 * I_lin(0), 2 * I_lin(1), 4 * I_mix(0), 6 * I_sqr(0), 2 * I_sqr(1),
		2 * I_lin(0), 4 * I_lin(1), 4 * I_mix(0), 2 * I_sqr(0), 6 * I_sqr(1);
	Eigen::FullPivLU<Eigen::Matrix<double, 5, 5>> lu(M);

	show_matrix_stats(M);

	// Compute L
	C.resize(5, num_kernels + 1 + dim + dim * (dim + 1) / 2);
	C.setZero();
	C.block(0, 0, dim, num_kernels) = K_lin.transpose();
	C.block(dim, 0, 1, num_kernels) = K_mix.transpose().colwise().sum();
	C.block(dim + 1, 0, dim, num_kernels) = 2.0 * (K_sqr.colwise() + K_cst).transpose();
	C.block(dim + 1, num_kernels, dim, 1).setConstant(2.0 * volume);
	C.bottomRightCorner(5, 5) = M;
	// std::cout << L.bottomRightCorner(10, 10) << std::endl;
}

void RBFWithQuadraticLagrange::compute_constraints_matrix_2d(const LinearAssembler &assembler,
															 const int num_bases, const Quadrature &quadr, Eigen::MatrixXd &C) const
{
	const int num_kernels = centers_.rows();
	const int space_dim = centers_.cols();
	const int assembler_dim = assembler.is_tensor() ? 2 : 1;
	assert(space_dim == 2);

	std::array<Eigen::MatrixXd, 5> strong;

	// ass_val = [q_10, q_01, q_11, q_20, q_02, psi_0, ..., psi_k]
	ElementAssemblyValues ass_val;
	ass_val.has_parameterization = false;
	ass_val.basis_values.resize(5 + num_kernels);

	// evaluating monomial and grad of monomials at quad points
	RBFWithQuadratic::setup_monomials_vals_2d(0, quadr.points, ass_val);
	RBFWithQuadratic::setup_monomials_strong_2d(assembler_dim, assembler, quadr.points, quadr.weights.array(), strong);

	// evaluating psi and grad psi at quadr points
	for (int j = 0; j < num_kernels; ++j)
	{
		ass_val.basis_values[5 + j].val = Eigen::MatrixXd(quadr.points.rows(), 1);
		ass_val.basis_values[5 + j].grad = Eigen::MatrixXd(quadr.points.rows(), quadr.points.cols());

		for (int q = 0; q < quadr.points.rows(); ++q)
		{
			const RowVectorNd p = quadr.points.row(q) - centers_.row(j);
			const double r = p.norm();

			ass_val.basis_values[5 + j].val(q) = kernel(is_volume(), r);
			ass_val.basis_values[5 + j].grad.row(q) = p * kernel_prime(is_volume(), r) / r;
		}
	}

	for (size_t i = 5; i < ass_val.basis_values.size(); ++i)
	{
		ass_val.basis_values[i].grad_t_m = ass_val.basis_values[i].grad;
	}

	// Compute C
	C.resize(RBFWithQuadratic::index_mapping(assembler_dim - 1, assembler_dim - 1, 4, assembler_dim) + 1, num_kernels + 1 + 5);
	C.setZero();

	for (int d = 0; d < 5; ++d)
	{
		// first num_kernels bases
		for (int i = 0; i < num_kernels; ++i)
		{
			const auto tmp = assembler.assemble(LinearAssemblerData(ass_val, d, 5 + i, quadr.weights));
			for (int alpha = 0; alpha < assembler_dim; ++alpha)
			{
				for (int beta = 0; beta < assembler_dim; ++beta)
				{
					const int loc_index = alpha * assembler_dim + beta;
					C(RBFWithQuadratic::index_mapping(alpha, beta, d, assembler_dim), i) = tmp(loc_index) + (strong[d].row(loc_index).transpose().array() * ass_val.basis_values[5 + i].val.array()).sum();
				}
			}
		}

		// second the q_i
		for (int i = 0; i < 5; ++i)
		{
			const auto tmp = assembler.assemble(LinearAssemblerData(ass_val, d, i, quadr.weights));
			for (int alpha = 0; alpha < assembler_dim; ++alpha)
			{
				for (int beta = 0; beta < assembler_dim; ++beta)
				{
					const int loc_index = alpha * assembler_dim + beta;
					C(RBFWithQuadratic::index_mapping(alpha, beta, d, assembler_dim), num_kernels + i + 1) = tmp(loc_index) + (strong[d].row(loc_index).transpose().array() * ass_val.basis_values[i].val.array()).sum();
				}
			}
		}

		// finally the constant
		for (int alpha = 0; alpha < assembler_dim; ++alpha)
		{
			for (int beta = 0; beta < assembler_dim; ++beta)
			{
				// std::cout<<alpha <<" "<< beta <<" "<< d <<" "<< assembler_dim<< " -> r = "<< RBFWithQuadratic::index_mapping(alpha, beta, d, assembler_dim)<<std::endl;
				const int loc_index = alpha * assembler_dim + beta;
				C(RBFWithQuadratic::index_mapping(alpha, beta, d, assembler_dim), num_kernels) = strong[d].row(loc_index).sum();
			}
		}
	}

	// {
	// 	std::ofstream file;
	// 	file.open("C.txt");
	// 	file << C;
	// 	file.close();
	// }

	// Eigen::MatrixXd Cold;
	// compute_constraints_matrix_2d_old(num_bases, quadr,Cold);
	// {
	// 	std::ofstream file;
	// 	file.open("Cold.txt");
	// 	file << Cold;
	// 	file.close();
	// }
}

// -----------------------------------------------------------------------------

void RBFWithQuadraticLagrange::compute_constraints_matrix_3d(
	const int num_bases, const Quadrature &quadr, Eigen::MatrixXd &C) const
{
	const int num_kernels = centers_.rows();
	const int dim = centers_.cols();
	assert(dim == 3);

	// K_cst = ∫φj
	// K_lin = ∫∇x(φj), ∫∇y(φj), ∫∇z(φj)
	// K_mix = ∫(y·∇x(φj)+x·∇y(φj)), ∫(z·∇y(φj)+y·∇z(φj)), ∫(x·∇z(φj)+z·∇x(φj))
	// K_sqr = ∫x·∇x(φj), ∫y·∇y(φj), ∫z·∇z(φj)
	Eigen::VectorXd K_cst = Eigen::VectorXd::Zero(num_kernels);
	Eigen::MatrixXd K_lin = Eigen::MatrixXd::Zero(num_kernels, dim);
	Eigen::MatrixXd K_mix = Eigen::MatrixXd::Zero(num_kernels, dim);
	Eigen::MatrixXd K_sqr = Eigen::MatrixXd::Zero(num_kernels, dim);
	for (int j = 0; j < num_kernels; ++j)
	{
		// ∫∇x(φj)(p) = Σ_q (xq - xk) * 1/r * h'(r) * wq
		// - xq is the x coordinate of the q-th quadrature point
		// - wq is the q-th quadrature weight
		// - r is the distance from pq to the kernel center
		// - h is the RBF kernel (scalar function)
		for (int q = 0; q < quadr.points.rows(); ++q)
		{
			const RowVectorNd p = quadr.points.row(q) - centers_.row(j);
			const double r = p.norm();
			const RowVectorNd gradPhi = p * kernel_prime(is_volume(), r) / r * quadr.weights(q);
			K_cst(j) += kernel(is_volume(), r) * quadr.weights(q);
			K_lin.row(j) += gradPhi;
			for (int d = 0; d < dim; ++d)
			{
				K_mix(j, d) += quadr.points(q, (d + 1) % dim) * gradPhi(d) + quadr.points(q, d) * gradPhi((d + 1) % dim);
			}
			K_sqr.row(j) += (quadr.points.row(q).array() * gradPhi.array()).matrix();
		}
	}

	// I_lin = ∫x, ∫y, ∫z
	// I_sqr = ∫x², ∫y², ∫z²
	// I_mix = ∫xy, ∫yz, ∫zx
	Eigen::RowVectorXd I_lin = (quadr.points.array().colwise() * quadr.weights.array()).colwise().sum();
	Eigen::RowVectorXd I_sqr = (quadr.points.array().square().colwise() * quadr.weights.array()).colwise().sum();
	Eigen::RowVectorXd I_mix(3);
	I_mix(0) = (quadr.points.col(0).array() * quadr.points.col(1).array() * quadr.weights.array()).sum();
	I_mix(1) = (quadr.points.col(1).array() * quadr.points.col(2).array() * quadr.weights.array()).sum();
	I_mix(2) = (quadr.points.col(2).array() * quadr.points.col(0).array() * quadr.weights.array()).sum();
	double volume = quadr.weights.sum();

	// std::cout << I_lin << std::endl;
	// std::cout << I_mix << std::endl;
	// std::cout << I_sqr << std::endl;

	// Compute M
	Eigen::Matrix<double, 9, 9> M;
	M << volume, 0, 0, I_lin(1), 0, I_lin(2), 2 * I_lin(0), 0, 0,
		0, volume, 0, I_lin(0), I_lin(2), 0, 0, 2 * I_lin(1), 0,
		0, 0, volume, 0, I_lin(1), I_lin(0), 0, 0, 2 * I_lin(2),
		I_lin(1), I_lin(0), 0, I_sqr(0) + I_sqr(1), I_mix(2), I_mix(1), 2 * I_mix(0), 2 * I_mix(0), 0,
		0, I_lin(2), I_lin(1), I_mix(2), I_sqr(1) + I_sqr(2), I_mix(0), 0, 2 * I_mix(1), 2 * I_mix(1),
		I_lin(2), 0, I_lin(0), I_mix(1), I_mix(0), I_sqr(2) + I_sqr(0), 2 * I_mix(2), 0, 2 * I_mix(2),
		2 * I_lin(0), 0, 0, 2 * I_mix(0), 0, 2 * I_mix(2), 4 * I_sqr(0), 0, 0,
		0, 2 * I_lin(1), 0, 2 * I_mix(0), 2 * I_mix(1), 0, 0, 4 * I_sqr(1), 0,
		0, 0, 2 * I_lin(2), 0, 2 * I_mix(1), 2 * I_mix(2), 0, 0, 4 * I_sqr(2);
	Eigen::Matrix<double, 1, 9> M_rhs;
	M_rhs.segment<3>(0) = I_lin;
	M_rhs.segment<3>(3) = I_mix;
	M_rhs.segment<3>(6) = I_sqr;
	// M_rhs << I_lin, I_mix, I_sqr;
	M.bottomRows(dim).rowwise() += 2.0 * M_rhs;
	// TODO
	//  assert(false);

	show_matrix_stats(M);

	// Compute L
	C.resize(9, num_kernels + 1 + dim + dim * (dim + 1) / 2);
	C.setZero();
	C.block(0, 0, dim, num_kernels) = K_lin.transpose();
	C.block(dim, 0, dim, num_kernels) = K_mix.transpose();
	C.block(dim + dim, 0, dim, num_kernels) = 2.0 * (K_sqr.colwise() + K_cst).transpose();
	C.block(dim + dim, num_kernels, dim, 1).setConstant(2.0 * volume);
	C.bottomRightCorner(9, 9) = M;
	// std::cout << C.bottomRightCorner(9, 12) << std::endl;
}

// -----------------------------------------------------------------------------

void RBFWithQuadraticLagrange::compute_weights(const LinearAssembler &assembler, const Eigen::MatrixXd &samples,
											   const Eigen::MatrixXd &local_basis_integral, const Quadrature &quadr,
											   Eigen::MatrixXd &b, bool with_constraints)
{
	logger().trace("#kernel centers: {}", centers_.rows());
	logger().trace("#collocation points: {}", samples.rows());
	logger().trace("#quadrature points: {}", quadr.weights.size());
	logger().trace("#non-vanishing bases: {}", b.cols());
	logger().trace("#constraints: {}", b.cols());

	// Compute A
	Eigen::MatrixXd A;
	compute_kernels_matrix(samples, A);

	if (!with_constraints)
	{
		// Solve the system
		const int num_kernels = centers_.rows();
		logger().trace("-- Solving system of size {}x{}", num_kernels, num_kernels);
		weights_ = (A.transpose() * A).ldlt().solve(A.transpose() * b);
		logger().trace("-- Solved!");

		return;
	}

	const int num_bases = b.cols();

	const Eigen::MatrixXd At = A.transpose();

	// Compute C
	Eigen::MatrixXd C;
	if (is_volume())
	{
		compute_constraints_matrix_3d(num_bases, quadr, C);
	}
	else
	{
		compute_constraints_matrix_2d(assembler, num_bases, quadr, C);
	}

	const int dim = centers_.cols();
	assert(centers_.rows() + 1 + dim + dim * (dim + 1) / 2 > C.rows());
	logger().trace("#constraints: {}", C.rows());

	// Compute rhs = [ A^T b; d ]
	assert(local_basis_integral.cols() == C.rows());
	assert(local_basis_integral.rows() == b.cols());
	assert(A.rows() == b.rows());
	Eigen::MatrixXd rhs(A.cols() + local_basis_integral.cols(), b.cols());
	rhs.topRows(A.cols()) = At * b;
	rhs.bottomRows(local_basis_integral.cols()) = local_basis_integral.transpose();

	// Compute M = [ A^T A, C^T; C, 0]
	assert(C.cols() == A.cols());
	assert(A.rows() == b.rows());
	Eigen::MatrixXd M(A.cols() + C.rows(), A.cols() + C.rows());
	M.topLeftCorner(A.cols(), A.cols()) = At * A;
	M.topRightCorner(A.cols(), C.rows()) = C.transpose();
	M.bottomLeftCorner(C.rows(), A.cols()) = C;
	M.bottomRightCorner(C.rows(), C.rows()).setZero();

	// std::cout << M.bottomRightCorner(10, 10) << std::endl;

	// Solve the system
	logger().trace("-- Solving system of size {}x{}", M.rows(), M.cols());
	auto ldlt = M.ldlt();
	if (ldlt.info() == Eigen::NumericalIssue)
	{
		logger().error("-- WARNING: Numerical issues when solving the harmonic least square.");
	}
	const auto tmp = ldlt.solve(rhs);
	weights_ = tmp.topRows(A.cols());
	logger().trace("-- Solved!");
	logger().trace("-- Mean residual: {}", (A * weights_ - b).array().abs().colwise().maxCoeff().mean());
	logger().trace("-- Max constraints error: {}", (C * weights_ - local_basis_integral.transpose()).array().abs().maxCoeff());

	//    {
	// 	std::ofstream file;
	// 	file.open("M.txt");
	// 	file << M;
	// 	file.close();
	// }

	// {
	// 	std::ofstream file;
	// 	file.open("C.txt");
	// 	file << C;
	// 	file.close();
	// }

	// {
	// 	std::ofstream file;
	// 	file.open("b.txt");
	// 	file << local_basis_integral.transpose();
	// 	file.close();
	// }

	//    {
	// 	std::ofstream file;
	// 	file.open("rhs.txt");
	// 	file << rhs;
	// 	file.close();
	// }

	// std::cout << M.bottomRightCorner(10, 10) << std::endl;

#if 0
	// Compute rhs = [ A^T b; d ]
	assert(local_basis_integral.cols() == C.rows());
	assert(local_basis_integral.rows() == b.cols());
	assert(A.rows() == b.rows());
	Eigen::MatrixXd rhs(A.cols() + local_basis_integral.cols(), b.cols());
	rhs.topRows(A.cols()) = At * b;
	rhs.bottomRows(local_basis_integral.cols()) = local_basis_integral.transpose();

	// Compute M = [ A^T A, C^T; C, 0]
	assert(C.cols() == A.cols());
	assert(A.rows() == b.rows());
	Eigen::MatrixXd M(A.cols() + C.rows(), A.cols() + C.rows());
	M.topLeftCorner(A.cols(), A.cols()) = At * A;
	M.topRightCorner(A.cols(), C.rows()) = C.transpose();
	M.bottomLeftCorner(C.rows(), A.cols()) = C;
	M.bottomRightCorner(C.rows(), C.rows()).setZero();

	// std::cout << M.bottomRightCorner(10, 10) << std::endl;

	// Solve the system
	logger().trace("-- Solving system of size {}x{}", M.rows(), M.cols());
	auto ldlt = M.ldlt();
	if (ldlt.info() == Eigen::NumericalIssue) {
		logger().error("-- WARNING: Numerical issues when solving the harmonic least square.");
	}
	weights_ = ldlt.solve(rhs).topRows(A.cols());
	logger().trace("-- Solved!");
    logger().trace("-- Mean residual: {}", (A * weights_ - b).array().abs().colwise().maxCoeff().mean());


	Eigen::MatrixXd MM, x, dx, val;
	basis(0, quadr.points, val);
	grad(0, quadr.points, MM);
	int dim = (is_volume() ? 3 : 2);
	for (int d = 0; d < dim; ++d) {
		// basis(0, quadr.points, x);
		// auto asd = quadr.points;
		// asd.col(d).array() += 1e-7;
		// basis(0, asd, dx);
		// std::cout << (dx - x) / 1e-7 - MM.col(d) << std::endl;
		std::cout << (MM.col(d).array() * quadr.weights.array()).sum() - local_basis_integral(0, d) << std::endl;
		std::cout << ((
				MM.col((d+1)%dim).array() * quadr.points.col(d).array()
				+ MM.col(d).array() * quadr.points.col((d+1)%dim).array()
			) * quadr.weights.array()).sum() - local_basis_integral(0, (dim == 2 ? 2 : (dim+d) )) << std::endl;
		std::cout << 2.0 * (
				(quadr.points.col(d).array() * MM.col(d).array()
				+ val.array())
			* quadr.weights.array()
			).sum() - local_basis_integral(0, (dim == 2 ? (3 + d) : (dim+dim+d))) << std::endl;
	}
#endif
}

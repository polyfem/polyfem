#include "Harmonic.hpp"
#include "PolygonQuadrature.hpp"
#include "Types.hpp"
#include <igl/Timer.h>
#include <iostream>
#include <fstream>

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

	Harmonic::Harmonic(const Eigen::MatrixXd &centers, const Eigen::MatrixXd &samples,
		const Eigen::MatrixXd &local_basis_integral, const Quadrature &quadr, Eigen::MatrixXd &rhs)
	: centers_(centers)
	{
		compute(samples, local_basis_integral, quadr, rhs);
	}

	void Harmonic::basis(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
	{
		const Eigen::MatrixXd w = weights_.col(local_index);
		const long end = w.size()-1;

		val.resize(uv.rows(), 1);
		val.setConstant(w(end)); //constant term

		for(long i = 0; i < uv.rows(); ++i)
		{
			const Eigen::MatrixXd p = uv.row(i);
			for(long k = 0; k < centers_.rows(); ++k)
				val(i) += w(k) * kernel(is_volume_, (centers_.row(k)-p).norm());

			//linear term
			if(is_volume_)
				val(i) += p(0) * w(end - 3) + p(1) * w(end - 2) + p(2) * w(end - 1);
			else
				val(i) += p(0) * w(end - 2) + p(1) * w(end - 1);
		}
	}

	void Harmonic::grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) const
	{
		const Eigen::MatrixXd w = weights_.col(local_index);
		const long end = w.size()-1;

		if(is_volume_)
			val.resize(uv.rows(), 3);
		else
			val.resize(uv.rows(), 2);

		for(long i = 0; i < uv.rows(); ++i)
		{
			if(is_volume_)
			{
				val(i, 0) = w(end-3);
				val(i, 1) = w(end-2);
				val(i, 2) = w(end-1);
			}
			else
			{
				val(i, 0) = w(end-2);
				val(i, 1) = w(end-1);
			}

			const auto &p = uv.row(i);
			for(long k = 0; k < centers_.rows(); ++k)
			{
				const double r = (centers_.row(k)-p).norm();

				val(i, 0) += w(k)*(p(0)-centers_(k,0))*kernel_prime(is_volume_, r)/r;
				val(i, 1) += w(k)*(p(1)-centers_(k,1))*kernel_prime(is_volume_, r)/r;

				if(is_volume_)
					val(i, 2) += w(k)*(p(2)-centers_(k,2))*kernel_prime(is_volume_, r)/r;
			}
		}
	}

	double sparsity(const Eigen::MatrixXd &M) {
		long cnt = 0;
		for (long i = 0; i < M.size(); ++i) {
			if (std::fabs(M.data()[i]) > 1e-7) {
				++cnt;
			}
		}
		return 100.0 * cnt / M.size();
	}

	void Harmonic::compute(const Eigen::MatrixXd &samples, const Eigen::MatrixXd &local_basis_integral,
		const Quadrature &quadr, Eigen::MatrixXd &rhs)
	{
		is_volume_ = samples.cols() == 3;

		std::cout << "#kernel centers: " << centers_.rows() << std::endl;
		std::cout << "#collocation points: " << samples.rows() << std::endl;
		std::cout << "#non-vanishing bases: " << rhs.cols() << std::endl;

#if 0
		const int dim = samples.cols();
		const int size = (int) samples.rows();

		//+dim linear, +1 constant
		Eigen::MatrixXd mat(size, centers_.rows() + dim + 1);

		const int end = int(mat.cols())-1;
		mat.col(end).setOnes();

		igl::Timer timer; timer.start();

		for(long i = 0; i < samples.rows(); ++i)
		{
			if (is_volume_) {
				mat(i, end - 3) = samples(i, 0);
				mat(i, end - 2) = samples(i, 1);
				mat(i, end - 1) = samples(i, 2);
			} else {
				mat(i, end - 2) = samples(i, 0);
				mat(i, end - 1) = samples(i, 1);
			}

			for(long j = 0; j < centers_.rows(); ++j)
			{
				mat(i,j)=kernel(is_volume_, (centers_.row(j)-samples.row(i)).norm());
			}
		}
		timer.stop();
		std::cout << "-- matrix A computed, took: " << timer.getElapsedTime() << std::endl;


		// Compute A
		// igl::Timer timer; timer.start();
		const int num_kernels = centers_.rows();
		Eigen::MatrixXd A(samples.rows(), num_kernels + dim + 1);
		for (int j = 0; j < num_kernels; ++j) {
			A.col(j) = (samples.rowwise() - centers_.row(j)).rowwise().norm().unaryExpr([this](double x)
				{ return kernel(is_volume_, x); });
		}
		A.middleCols(num_kernels, dim) = samples;
		A.rightCols<1>().setOnes();
		// timer.stop();
		std::cout << "-- Computed A: " << timer.getElapsedTime() << std::endl;

		std::cout << "-- A diff: " << (A-mat).norm() << std::endl;


		std::cout << "solving system of size " << centers_.rows() << " x " << centers_.rows() << std::endl;
		// weights_ = mat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
		weights_ = (mat.transpose() * mat).ldlt().solve(mat.transpose() * rhs);
		std::cout << "done" << std::endl;
#endif
#if 0
		// For each basis function f that is nonzero on the element E, we want to
		// solve the least square system A w = rhs, where:
		//     ┏                    ┓
		//     ┃ φj(pi) ... xi yi 1 ┃
		// A = ┃   ┊        ┊  ┊  ┊ ┃ ∊ ℝ^{#S x (#K+dim+1)}
		//     ┃   ┊        ┊  ┊  ┊ ┃
		//     ┗                    ┛
		//     ┏                ┓^⊤
		// w = ┃ wj ... ax ay c ┃   ∊ ℝ^{#K+dim+1}
		//     ┗                ┛
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
		// t = ┃ 0  ┈  ┈  0 lbx lby 0 ┃   / Vol(E) ∊ ℝ^{#K+dim+1}
		//     ┗                      ┛
		//
		//     ┏                  ┓
		//     ┃   1              ┃
		//     ┃       1          ┃
		//     ┃          ·       ┃
		// L = ┃             ·    ┃ ∊ ℝ^{ (#K+dim+1) x (#K+1}) }
		//     ┃ Lx_j  ┈        0 ┃
		//     ┃ Ly_j  ┈        0 ┃
		//     ┃                1 ┃
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
			// - h is the harmonic RBF kernel (scalar function)
			const Eigen::MatrixXd drdp = quadr.points.rowwise() - centers_.row(j);
			const Eigen::VectorXd r = drdp.rowwise().norm();
			KI.row(j) = (drdp.array().colwise() * (quadr.weights.array() * r.unaryExpr([this](double x)
				{ return kernel_prime(is_volume_, x); }).array() / r.array())).colwise().sum();
		}
		KI /= quadr.weights.sum();

		// Compute L
		Eigen::MatrixXd L(num_kernels + dim + 1, num_kernels + 1);
		L.setZero();
		L.diagonal().setOnes();
		L.bottomRightCorner(dim+1, 1).setZero();
		L.bottomRightCorner(1, 1).setOnes();
		L.block(num_kernels, 0, dim, num_kernels) = -KI.transpose();
		// std::cout << L.bottomRightCorner(10, 10) << std::endl;

		// Compute A
		Eigen::MatrixXd A(samples.rows(), num_kernels + dim + 1);
		for (int j = 0; j < num_kernels; ++j) {
			A.col(j) = (samples.rowwise() - centers_.row(j)).rowwise().norm().unaryExpr([this](double x)
				{ return kernel(is_volume_, x); });
		}
		A.middleCols(num_kernels, dim) = samples;
		A.rightCols<1>().setOnes();

		// Compute t
		weights_.resize(num_kernels + dim + 1, num_bases);
		weights_.setZero();
		weights_.middleRows(num_kernels, dim) = local_basis_integral.transpose() / quadr.weights.sum();

		// Compute b = rhs - A t
		rhs -= A * weights_;

		// Solve the system
		std::cout << "-- Solving system of size " << num_kernels << " x " << num_kernels << std::endl;
		weights_ += L * (L.transpose() * A.transpose() * A * L).ldlt().solve(L.transpose() * A.transpose() * rhs);
		// Eigen::MatrixXd M = samples * local_basis_integral.transpose();
		// std::cout << "t: " << M.rows() << " x " << M.cols() << std::endl;
		// + samples * local_basis_integral.transpose();

		// weights_ = (A.transpose() * A).ldlt().solve(A.transpose() * rhs);

		// v = rhs;
		// rhs = rhs2;
		// weights_.setZero();
#else
		const int dim = samples.cols();
		const int size = (int) samples.rows();

		//+1 constant
		Eigen::MatrixXd mat(size, centers_.rows() + 1);

		const int end = int(mat.cols())-1;
		mat.col(end).setOnes();

		// igl::Timer timer; timer.start();
		Eigen::MatrixXd KI(centers_.rows(), dim);

		// Eigen::MatrixXd KI(centers_.rows(), dim);
		for(long j = 0; j < centers_.rows(); ++j)
		{
			// const double r = (centers_.row(j)-samples.row(i)).norm();
			const Eigen::MatrixXd diff_r_x = quadr.points.col(0).array() - centers_(j, 0);
			const Eigen::MatrixXd diff_r_y = quadr.points.col(1).array() - centers_(j, 1);
			Eigen::MatrixXd diff_r_z;
			if(is_volume_)
				diff_r_z = quadr.points.col(2).array() - centers_(j, 2);

			const Eigen::MatrixXd rr = is_volume_ ?
			(diff_r_x.array() * diff_r_x.array() + diff_r_y.array() * diff_r_y.array() + diff_r_z.array() * diff_r_z.array()).sqrt().eval() :
			(diff_r_x.array() * diff_r_x.array() + diff_r_y.array() * diff_r_y.array()).sqrt().eval();

			double KxI = 0;
			double KyI = 0;
			double KzI = 0;

			for(long k = 0; k < rr.size(); ++k)
			{
				KxI += diff_r_x(k) / rr(k) * kernel_prime(is_volume_, rr(k)) * quadr.weights(k);
				KyI += diff_r_y(k) / rr(k) * kernel_prime(is_volume_, rr(k)) * quadr.weights(k);

				if(is_volume_)
					KzI += diff_r_z(k) / rr(k) * kernel_prime(is_volume_, rr(k)) * quadr.weights(k);
			}

			if (is_volume_) { KI.row(j) << KxI, KyI, KzI; } else { KI.row(j) << KxI, KyI; }
			if (j == 0) { std::cout << KI.row(j) << std::endl; }
			KI.row(j) /= quadr.weights.sum();
		}

		// timer.stop();
		// std::cout << "-- KI computed, took: " << timer.getElapsedTime() << std::endl;
		// timer.start();

		for(long i = 0; i < samples.rows(); ++i)
		{
			const double x = samples(i, 0);
			const double y = samples(i, 1);
			const double z = is_volume_ ? samples(i, 2) : 0;
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> xyz(dim);
			if (is_volume_) {
				xyz << x, y, z;
			} else {
				xyz << x, y;
			}

			rhs.row(i) -= (local_basis_integral.col(0).transpose() * x + local_basis_integral.col(1).transpose() * y)/quadr.weights.sum();
			if(is_volume_)
				rhs.row(i) -= local_basis_integral.col(2).transpose() * z / quadr.weights.sum();

			for(long j = 0; j < centers_.rows(); ++j)
			{
				const double r = (centers_.row(j)-samples.row(i)).norm();
				mat(i,j)=kernel(is_volume_, r) - KI.row(j).dot(xyz);
			}
		}

		// timer.stop();
		// std::cout << "-- matrix A computed, took: " << timer.getElapsedTime() << std::endl;
		// timer.start();


		// Eigen::MatrixXd tmp = mat.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
		std::cout << "solving system of size " << centers_.rows() << " x " << centers_.rows() << std::endl;

		Eigen::MatrixXd tmp = (mat.transpose() * mat).ldlt().solve(mat.transpose() * rhs);
		// timer.stop();
		// std::cout << "-- solved, took: " << timer.getElapsedTime() << std::endl;
		// timer.start();
		weights_.resize(centers_.rows() + (is_volume_?3:2) + 1, tmp.cols());
		weights_.setZero();
		weights_.block(0, 0, centers_.rows(), tmp.cols()) = tmp.block(0, 0, centers_.rows(), tmp.cols());

		const int wend = weights_.rows()-1;

		weights_.row(wend) = tmp.row(tmp.rows()-1);
		if(is_volume_)
		{
			weights_.row(wend-3) = local_basis_integral.col(0).transpose()/quadr.weights.sum();
			weights_.row(wend-2) = local_basis_integral.col(1).transpose()/quadr.weights.sum();
			weights_.row(wend-1) = local_basis_integral.col(2).transpose()/quadr.weights.sum();
		}
		else
		{
			weights_.row(wend-2) = local_basis_integral.col(0).transpose()/quadr.weights.sum();
			weights_.row(wend-1) = local_basis_integral.col(1).transpose()/quadr.weights.sum();
		}

		for(long j = 0; j < centers_.rows(); ++j)
		{
			const Eigen::MatrixXd diff_r_x = quadr.points.col(0).array() - centers_(j, 0);
			const Eigen::MatrixXd diff_r_y = quadr.points.col(1).array() - centers_(j, 1);
			Eigen::MatrixXd diff_r_z;
			if(is_volume_)
				diff_r_z = quadr.points.col(2).array() - centers_(j, 2);

			const Eigen::MatrixXd rr = is_volume_ ?
			(diff_r_x.array() * diff_r_x.array() + diff_r_y.array() * diff_r_y.array() + diff_r_z.array() * diff_r_z.array()).sqrt().eval() :
			(diff_r_x.array() * diff_r_x.array() + diff_r_y.array() * diff_r_y.array()).sqrt().eval();

			double KxI = 0;
			double KyI = 0;
			double KzI = 0;

			for(long k = 0; k < rr.size(); ++k)
			{
				KxI += diff_r_x(k) / rr(k) * kernel_prime(is_volume_, rr(k)) * quadr.weights(k);
				KyI += diff_r_y(k) / rr(k) * kernel_prime(is_volume_, rr(k)) * quadr.weights(k);

				if(is_volume_)
					KzI += diff_r_z(k) / rr(k) * kernel_prime(is_volume_, rr(k)) * quadr.weights(k);
			}

			KxI /= quadr.weights.sum();
			KyI /= quadr.weights.sum();
			KzI /= quadr.weights.sum();

			if(is_volume_)
			{
				weights_.row(wend-3) -= weights_.row(j)*KxI;
				weights_.row(wend-2) -= weights_.row(j)*KyI;
				weights_.row(wend-1) -= weights_.row(j)*KzI;
			}
			else
			{
				weights_.row(wend-2) -= weights_.row(j)*KxI;
				weights_.row(wend-1) -= weights_.row(j)*KyI;
			}
		}

		// std::cout << "-- W diff: " << (weights_ - v).norm() << std::endl;

		// Eigen::MatrixXd integralx = Eigen::MatrixXd::Zero(weights_.cols(), 1);
		// Eigen::MatrixXd integraly = Eigen::MatrixXd::Zero(weights_.cols(), 1);
		// Eigen::MatrixXd gradv;
		// for(long asd = 0; asd < quadr.weights.rows(); ++asd)
		// {
		// 	for(long k = 0; k < weights_.cols(); ++k){
		// 		grad(k, quadr.points.row(asd), gradv);
		// 		integralx(k) += gradv(0)*quadr.weights(asd);
		// 		integraly(k) += gradv(1)*quadr.weights(asd);
		// 	}
		// }

		// std::cout<<integralx-local_basis_integral.col(0)<<"\n------------\n"<<std::endl;
		// std::cout<<integraly-local_basis_integral.col(1)<<"\n------------\n"<<std::endl;
		// std::cout<<local_basis_integral<<"\n------------\n"<<std::endl;

		// std::cout.precision(100);
		// // std::cout<<"mat=[\n"<<mat<<"];"<<std::endl;
		// std::cout<<"weights=[\n"<<tmp<<"];"<<std::endl;
		// std::cout<<"weights=[\n"<<weights_<<"];"<<std::endl;

		// {
		// 	std::ofstream os;
		// 	os.open("cc.txt");
		// 	os.precision(100);
		// 	os<<centers_<<std::endl;
		// 	os.close();
		// }

		// {
		// 	std::ofstream os;
		// 	os.open("ss.txt");
		// 	os.precision(100);
		// 	os<<samples<<std::endl;
		// 	os.close();
		// }

		// // {
		// // 	std::ofstream os;
		// // 	os.open("mat.txt");
		// // 	os.precision(100);
		// // 	os<<mat<<std::endl;
		// // 	os.close();
		// // }

		// {
		// 	std::ofstream os;
		// 	os.open("rr.txt");
		// 	os.precision(100);
		// 	os<<rhs<<std::endl;
		// 	os.close();
		// }

		// {
		// 	std::ofstream os;
		// 	os.open("ww.txt");
		// 	os.precision(100);
		// 	os<<weights_<<std::endl;
		// 	os.close();
		// }

		// std::cout << "#weights: " << weights_.rows() << " x " << weights_.cols() << std::endl;
		// timer.stop();
		// std::cout << "-- computed harmonic, took: " << timer.getElapsedTime() << std::endl;
#endif
	}
}

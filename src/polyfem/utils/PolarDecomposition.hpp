#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>

#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>

#include <Eigen/Dense>

namespace polyfem
{
	namespace utils
	{
		template <class scalar>
		scalar real_abs(const scalar &x)
		{
			return pow(pow(x, 2), 0.5);
		}

		// S = sqrt(A), A is SPD, compute dS/dA
		template <class scalar>
		Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_sqrt_grad(const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> &S)
		{
			const int dim = S.rows();
			assert(S.cols() == dim);
			Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> id = Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(dim, dim);

			return (Eigen::kroneckerProduct(S, id) + Eigen::kroneckerProduct(id, S)).inverse();
		}

		template <class scalar>
		void polar_decomposition(const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> &F, Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> &R, Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> &U)
		{
			const int dim = F.rows();
			assert(F.cols() == dim);

			if (dim == 2)
			{
				scalar detF = F.determinant();
				Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> adjFstar = (detF * F.inverse()).transpose();
				Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> M = F + (detF / real_abs(detF)) * adjFstar;
				U = (M.transpose() * F) / pow(real_abs(M.determinant()), 0.5);
			}
			else if (dim == 3)
			{
				
			}
			else
				throw std::runtime_error("Only support 2x2 or 3x3 matrix polar decomposition!");
		}

		void finite_diff_complex_step(const Eigen::VectorXd &x, const std::function<Eigen::VectorXcd(const Eigen::VectorXcd&)>& f, Eigen::MatrixXd& jac, const double eps)
		{
			using namespace std::complex_literals;

			jac.setZero(f(x).rows(), x.rows());

			Eigen::VectorXcd x_mutable = x;
			for (size_t i = 0; i < x.rows(); i++)
			{
				x_mutable[i] += eps * 1i;
				jac.col(i) = f(x_mutable).imag();
				x_mutable[i] = x[i];
			}

			jac /= eps;
		}

		template <class scalar>
		void polar_decomposition_grad(const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> &F, const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> &R, const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> &U, Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> &dUdF)
		{
			const int dim = F.rows();
			assert(F.cols() == dim);

			Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(dim*dim, 0, dim*dim-1).reshaped(dim, dim).reshaped<Eigen::RowMajor>();
			Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> tmp = Eigen::kroneckerProduct(F.transpose(), Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(dim, dim))(Eigen::all, ind);

			dUdF = matrix_sqrt_grad(U) * (tmp + tmp(ind, Eigen::all));
		}
    }
}
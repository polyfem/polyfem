#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/assembler/AssemblerData.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <Eigen/Dense>

#include <vector>
#include <array>
#include <functional>

namespace polyfem
{
	constexpr int SMALL_N = POLYFEM_SMALL_N;
	constexpr int BIG_N = POLYFEM_BIG_N;

	enum class ElasticityTensorType
	{
		CAUCHY,
		PK1,
		PK2,
		F
	};

	Eigen::VectorXd
	gradient_from_energy(const int size, const int n_bases, const assembler::NonLinearAssemblerData &data,
						 const std::function<DScalar1<double, Eigen::Matrix<double, 6, 1>>(const assembler::NonLinearAssemblerData &)> &fun6,
						 const std::function<DScalar1<double, Eigen::Matrix<double, 8, 1>>(const assembler::NonLinearAssemblerData &)> &fun8,
						 const std::function<DScalar1<double, Eigen::Matrix<double, 12, 1>>(const assembler::NonLinearAssemblerData &)> &fun12,
						 const std::function<DScalar1<double, Eigen::Matrix<double, 18, 1>>(const assembler::NonLinearAssemblerData &)> &fun18,
						 const std::function<DScalar1<double, Eigen::Matrix<double, 24, 1>>(const assembler::NonLinearAssemblerData &)> &fun24,
						 const std::function<DScalar1<double, Eigen::Matrix<double, 30, 1>>(const assembler::NonLinearAssemblerData &)> &fun30,
						 const std::function<DScalar1<double, Eigen::Matrix<double, 60, 1>>(const assembler::NonLinearAssemblerData &)> &fun60,
						 const std::function<DScalar1<double, Eigen::Matrix<double, 81, 1>>(const assembler::NonLinearAssemblerData &)> &fun81,
						 const std::function<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>>(const assembler::NonLinearAssemblerData &)> &funN,
						 const std::function<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 1000, 1>>(const assembler::NonLinearAssemblerData &)> &funBigN,
						 const std::function<DScalar1<double, Eigen::VectorXd>(const assembler::NonLinearAssemblerData &)> &funn);

	Eigen::MatrixXd hessian_from_energy(const int size, const int n_bases, const assembler::NonLinearAssemblerData &data,
										const std::function<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>(const assembler::NonLinearAssemblerData &)> &fun6,
										const std::function<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>(const assembler::NonLinearAssemblerData &)> &fun8,
										const std::function<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>(const assembler::NonLinearAssemblerData &)> &fun12,
										const std::function<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>(const assembler::NonLinearAssemblerData &)> &fun18,
										const std::function<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>(const assembler::NonLinearAssemblerData &)> &fun24,
										const std::function<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>(const assembler::NonLinearAssemblerData &)> &fun30,
										const std::function<DScalar2<double, Eigen::Matrix<double, 60, 1>, Eigen::Matrix<double, 60, 60>>(const assembler::NonLinearAssemblerData &)> &fun60,
										const std::function<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>(const assembler::NonLinearAssemblerData &)> &fun81,
										const std::function<DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, SMALL_N, SMALL_N>>(const assembler::NonLinearAssemblerData &)> &funN,
										const std::function<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>(const assembler::NonLinearAssemblerData &)> &funn);

	double von_mises_stress_for_stress_tensor(const Eigen::MatrixXd &stress);
	Eigen::MatrixXd pk1_from_cauchy(const Eigen::MatrixXd &stress, const Eigen::MatrixXd &F);
	Eigen::MatrixXd pk2_from_cauchy(const Eigen::MatrixXd &stress, const Eigen::MatrixXd &F);
	void compute_diplacement_grad(const int size, const assembler::ElementAssemblyValues &vals, const Eigen::MatrixXd &local_pts, const int p, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &displacement_grad);
	void compute_diplacement_grad(const int size, const basis::ElementBases &bs, const assembler::ElementAssemblyValues &vals, const Eigen::MatrixXd &local_pts, const int p, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &displacement_grad);

	double convert_to_lambda(const bool is_volume, const double E, const double nu);
	double convert_to_mu(const double E, const double nu);
	Eigen::Matrix2d d_lambda_mu_d_E_nu(const bool is_volume, const double E, const double nu);
	Eigen::Matrix2d d_E_nu_d_lambda_mu(const bool is_volume, const double lambda, const double mu);
	double convert_to_E(const bool is_volume, const double lambda, const double mu);
	double convert_to_nu(const bool is_volume, const double lambda, const double mu);

	typedef std::array<double, 2> DampingParameters;

	template <typename AutoDiffVect>
	void get_local_disp(
		const assembler::NonLinearAssemblerData &data,
		const int size,
		AutoDiffVect &local_disp)
	{
		typedef typename AutoDiffVect::Scalar T;

		assert(data.x.cols() == 1);

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_dispv(data.vals.basis_values.size() * size, 1);
		local_dispv.setZero();
		for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			for (size_t ii = 0; ii < bs.global.size(); ++ii)
			{
				for (int d = 0; d < size; ++d)
				{
					local_dispv(i * size + d) += bs.global[ii].val * data.x(bs.global[ii].index * size + d);
				}
			}
		}

		DiffScalarBase::setVariableCount(local_dispv.rows());
		local_disp.resize(local_dispv.rows(), 1);

		const AutoDiffAllocator<T> allocate_auto_diff_scalar;

		for (long i = 0; i < local_dispv.rows(); ++i)
		{
			local_disp(i) = allocate_auto_diff_scalar(i, local_dispv(i));
		}
	}

	template <typename AutoDiffVect, typename AutoDiffGradMat>
	void compute_disp_grad_at_quad(
		const assembler::NonLinearAssemblerData &data,
		const AutoDiffVect &local_disp,
		const int p, const int size,
		AutoDiffGradMat &def_grad)
	{
		typedef typename AutoDiffVect::Scalar T;

		for (long k = 0; k < def_grad.size(); ++k)
			def_grad(k) = T(0);

		for (size_t i = 0; i < data.vals.basis_values.size(); ++i)
		{
			const auto &bs = data.vals.basis_values[i];
			const Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> grad = bs.grad.row(p);
			assert(grad.size() == size);

			for (int d = 0; d < size; ++d)
			{
				for (int c = 0; c < size; ++c)
				{
					def_grad(d, c) += grad(c) * local_disp(i * size + d);
				}
			}
		}

		AutoDiffGradMat jac_it(size, size);
		for (long k = 0; k < jac_it.size(); ++k)
			jac_it(k) = T(data.vals.jac_it[p](k));
		def_grad = def_grad * jac_it;
	}

	// https://en.wikipedia.org/wiki/Invariants_of_tensors
	template <typename AutoDiffGradMat>
	typename AutoDiffGradMat::Scalar first_invariant(const AutoDiffGradMat &B)
	{
		return B.trace();
	}

	template <typename AutoDiffGradMat>
	typename AutoDiffGradMat::Scalar second_invariant(const AutoDiffGradMat &B)
	{
		return 0.5 * (B.trace() * B.trace() - (B * B).trace());
	}

	template <typename AutoDiffGradMat>
	typename AutoDiffGradMat::Scalar third_invariant(const AutoDiffGradMat &B)
	{
		return polyfem::utils::determinant(B);
	}
} // namespace polyfem

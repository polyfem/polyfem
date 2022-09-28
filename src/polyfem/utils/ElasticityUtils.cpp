#include "ElasticityUtils.hpp"
#include <polyfem/utils/Logger.hpp>

#include <tinyexpr.h>

namespace polyfem
{
	using namespace assembler;
	using namespace basis;
	using namespace utils;

	Eigen::VectorXd gradient_from_energy(const int size, const int n_bases, const assembler::NonLinearAssemblerData &data,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 6, 1>>(const assembler::NonLinearAssemblerData &)> &fun6,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 8, 1>>(const assembler::NonLinearAssemblerData &)> &fun8,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 12, 1>>(const assembler::NonLinearAssemblerData &)> &fun12,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 18, 1>>(const assembler::NonLinearAssemblerData &)> &fun18,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 24, 1>>(const assembler::NonLinearAssemblerData &)> &fun24,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 30, 1>>(const assembler::NonLinearAssemblerData &)> &fun30,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 60, 1>>(const assembler::NonLinearAssemblerData &)> &fun60,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 81, 1>>(const assembler::NonLinearAssemblerData &)> &fun81,
										 const std::function<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>>(const assembler::NonLinearAssemblerData &)> &funN,
										 const std::function<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, BIG_N, 1>>(const assembler::NonLinearAssemblerData &)> &funBigN,
										 const std::function<DScalar1<double, Eigen::VectorXd>(const assembler::NonLinearAssemblerData &)> &funn)
	{
		Eigen::VectorXd grad;

		switch (size * n_bases)
		{
		case 6:
		{
			auto auto_diff_energy = fun6(data);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 8:
		{
			auto auto_diff_energy = fun8(data);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 12:
		{
			auto auto_diff_energy = fun12(data);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 18:
		{
			auto auto_diff_energy = fun18(data);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 24:
		{
			auto auto_diff_energy = fun24(data);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 30:
		{
			auto auto_diff_energy = fun30(data);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 60:
		{
			auto auto_diff_energy = fun60(data);
			grad = auto_diff_energy.getGradient();
			break;
		}
		case 81:
		{
			auto auto_diff_energy = fun81(data);
			grad = auto_diff_energy.getGradient();
			break;
		}
		}

		if (grad.size() <= 0)
		{
			if (n_bases * size <= SMALL_N)
			{
				auto auto_diff_energy = funN(data);
				grad = auto_diff_energy.getGradient();
			}
			else if (n_bases * size <= BIG_N)
			{
				auto auto_diff_energy = funBigN(data);
				grad = auto_diff_energy.getGradient();
			}
			else
			{
				static bool show_message = true;

				if (show_message)
				{
					logger().debug("[Warning] grad {}^{} not using static sizes", n_bases, size);
					show_message = false;
				}

				auto auto_diff_energy = funn(data);
				grad = auto_diff_energy.getGradient();
			}
		}

		return grad;
	}

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
										const std::function<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>(const assembler::NonLinearAssemblerData &)> &funn)
	{
		Eigen::MatrixXd hessian;

		switch (size * n_bases)
		{
		case 6:
		{
			auto auto_diff_energy = fun6(data);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 8:
		{
			auto auto_diff_energy = fun8(data);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 12:
		{
			auto auto_diff_energy = fun12(data);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 18:
		{
			auto auto_diff_energy = fun18(data);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 24:
		{
			auto auto_diff_energy = fun24(data);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 30:
		{
			auto auto_diff_energy = fun30(data);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 60:
		{
			auto auto_diff_energy = fun60(data);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		case 81:
		{
			auto auto_diff_energy = fun81(data);
			hessian = auto_diff_energy.getHessian();
			break;
		}
		}

		if (hessian.size() <= 0)
		{
			if (n_bases * size <= SMALL_N)
			{
				auto auto_diff_energy = funN(data);
				hessian = auto_diff_energy.getHessian();
			}
			else
			{
				static bool show_message = true;

				if (show_message)
				{
					logger().debug("[Warning] hessian {}*{} not using static sizes", n_bases, size);
					show_message = false;
				}

				auto auto_diff_energy = funn(data);
				hessian = auto_diff_energy.getHessian();
			}
		}

		// time.stop();
		// std::cout << "-- hessian: " << time.getElapsedTime() << std::endl;

		return hessian;
	}

	double convert_to_lambda(const bool is_volume, const double E, const double nu)
	{
		if (is_volume)
			return (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));

		return (nu * E) / (1.0 - nu * nu);
	}

	double convert_to_mu(const double E, const double nu)
	{
		return E / (2.0 * (1.0 + nu));
	}

	void compute_diplacement_grad(const int size, const ElementBases &bs, const ElementAssemblyValues &vals, const Eigen::MatrixXd &local_pts, const int p, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &displacement_grad)
	{
		assert(displacement.cols() == 1);

		displacement_grad.setZero();

		for (std::size_t j = 0; j < bs.bases.size(); ++j)
		{
			const Basis &b = bs.bases[j];
			const auto &loc_val = vals.basis_values[j];

			assert(bs.bases.size() == vals.basis_values.size());
			assert(loc_val.grad.rows() == local_pts.rows());
			assert(loc_val.grad.cols() == size);

			for (int d = 0; d < size; ++d)
			{
				for (std::size_t ii = 0; ii < b.global().size(); ++ii)
				{
					displacement_grad.row(d) += b.global()[ii].val * loc_val.grad.row(p) * displacement(b.global()[ii].index * size + d);
				}
			}
		}

		displacement_grad = (displacement_grad * vals.jac_it[p]).eval();
	}

	double von_mises_stress_for_stress_tensor(const Eigen::MatrixXd &stress)
	{
		double von_mises_stress;

		if (stress.rows() == 3)
		{
			von_mises_stress = 0.5 * (stress(0, 0) - stress(1, 1)) * (stress(0, 0) - stress(1, 1)) + 3.0 * stress(0, 1) * stress(1, 0);
			von_mises_stress += 0.5 * (stress(2, 2) - stress(1, 1)) * (stress(2, 2) - stress(1, 1)) + 3.0 * stress(2, 1) * stress(2, 1);
			von_mises_stress += 0.5 * (stress(2, 2) - stress(0, 0)) * (stress(2, 2) - stress(0, 0)) + 3.0 * stress(2, 0) * stress(2, 0);
		}
		else
		{
			// von_mises_stress = ( stress(0, 0) - stress(1, 1) ) * ( stress(0, 0) - stress(1, 1) ) + 3.0  *  stress(0, 1) * stress(1, 0);
			von_mises_stress = stress(0, 0) * stress(0, 0) - stress(0, 0) * stress(1, 1) + stress(1, 1) * stress(1, 1) + 3.0 * stress(0, 1) * stress(1, 0);
		}

		von_mises_stress = sqrt(fabs(von_mises_stress));

		return von_mises_stress;
	}

	void ElasticityTensor::resize(const int size)
	{
		if (size == 2)
			stifness_tensor_.resize(3, 3);
		else
			stifness_tensor_.resize(6, 6);

		stifness_tensor_.setZero();

		size_ = size;
	}

	double ElasticityTensor::operator()(int i, int j) const
	{
		if (j < i)
		{
			std::swap(i, j);
		}

		assert(j >= i);
		return stifness_tensor_(i, j);
	}

	double &ElasticityTensor::operator()(int i, int j)
	{
		if (j < i)
		{
			std::swap(i, j);
		}

		assert(j >= i);
		return stifness_tensor_(i, j);
	}

	void ElasticityTensor::set_from_entries(const std::vector<double> &entries)
	{
		if (size_ == 2)
		{
			if (entries.size() == 4)
			{
				set_orthotropic(
					entries[0],
					entries[1],
					entries[2],
					entries[3]);

				return;
			}

			assert(entries.size() >= 6);

			(*this)(0, 0) = entries[0];
			(*this)(0, 1) = entries[1];
			(*this)(0, 2) = entries[2];

			(*this)(1, 1) = entries[3];
			(*this)(1, 2) = entries[4];

			(*this)(2, 2) = entries[5];
		}
		else
		{
			if (entries.size() == 9)
			{
				set_orthotropic(
					entries[0],
					entries[1],
					entries[2],
					entries[3],
					entries[4],
					entries[5],
					entries[6],
					entries[7],
					entries[8]);

				return;
			}
			assert(entries.size() >= 21);

			(*this)(0, 0) = entries[0];
			(*this)(0, 1) = entries[1];
			(*this)(0, 2) = entries[2];
			(*this)(0, 3) = entries[3];
			(*this)(0, 4) = entries[4];
			(*this)(0, 5) = entries[5];

			(*this)(1, 1) = entries[6];
			(*this)(1, 2) = entries[7];
			(*this)(1, 3) = entries[8];
			(*this)(1, 4) = entries[9];
			(*this)(1, 5) = entries[10];

			(*this)(2, 2) = entries[11];
			(*this)(2, 3) = entries[12];
			(*this)(2, 4) = entries[13];
			(*this)(2, 5) = entries[14];

			(*this)(3, 3) = entries[15];
			(*this)(3, 4) = entries[16];
			(*this)(3, 5) = entries[17];

			(*this)(4, 4) = entries[18];
			(*this)(4, 5) = entries[19];

			(*this)(5, 5) = entries[20];
		}
	}

	void ElasticityTensor::set_from_lambda_mu(const double lambda, const double mu)
	{
		if (size_ == 2)
		{
			(*this)(0, 0) = 2 * mu + lambda;
			(*this)(0, 1) = lambda;
			(*this)(0, 2) = 0;

			(*this)(1, 1) = 2 * mu + lambda;
			(*this)(1, 2) = 0;

			(*this)(2, 2) = mu;
		}
		else
		{
			(*this)(0, 0) = 2 * mu + lambda;
			(*this)(0, 1) = lambda;
			(*this)(0, 2) = lambda;
			(*this)(0, 3) = 0;
			(*this)(0, 4) = 0;
			(*this)(0, 5) = 0;

			(*this)(1, 1) = 2 * mu + lambda;
			(*this)(1, 2) = lambda;
			(*this)(1, 3) = 0;
			(*this)(1, 4) = 0;
			(*this)(1, 5) = 0;

			(*this)(2, 2) = 2 * mu + lambda;
			(*this)(2, 3) = 0;
			(*this)(2, 4) = 0;
			(*this)(2, 5) = 0;

			(*this)(3, 3) = mu;
			(*this)(3, 4) = 0;
			(*this)(3, 5) = 0;

			(*this)(4, 4) = mu;
			(*this)(4, 5) = 0;

			(*this)(5, 5) = mu;
		}
	}

	void ElasticityTensor::set_from_young_poisson(const double young, const double nu)
	{
		if (size_ == 2)
		{
			stifness_tensor_ << 1.0, nu, 0.0,
				nu, 1.0, 0.0,
				0.0, 0.0, (1.0 - nu) / 2.0;
			stifness_tensor_ *= young / (1.0 - nu * nu);
		}
		else
		{
			assert(size_ == 3);
			const double v = nu;
			stifness_tensor_ << 1. - v, v, v, 0, 0, 0,
				v, 1. - v, v, 0, 0, 0,
				v, v, 1. - v, 0, 0, 0,
				0, 0, 0, (1. - 2. * v) / 2., 0, 0,
				0, 0, 0, 0, (1. - 2. * v) / 2., 0,
				0, 0, 0, 0, 0, (1. - 2. * v) / 2.;
			stifness_tensor_ *= young / ((1. + v) * (1. - 2. * v));
		}
	}

	void ElasticityTensor::set_orthotropic(
		double Ex, double Ey, double Ez,
		double nuYX, double nuZX, double nuZY,
		double muYZ, double muZX, double muXY)
	{
		// copied from Julian
		assert(size_ == 3);
		// Note: this isn't the flattened compliance tensor! Rather, it is the
		// matrix inverse of the flattened elasticity tensor. See the tensor
		// flattening writeup.
		stifness_tensor_ << 1.0 / Ex, -nuYX / Ey, -nuZX / Ez, 0.0, 0.0, 0.0,
			0.0, 1.0 / Ey, -nuZY / Ez, 0.0, 0.0, 0.0,
			0.0, 0.0, 1.0 / Ez, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 1.0 / muYZ, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 1.0 / muZX, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / muXY;
	}

	void ElasticityTensor::set_orthotropic(double Ex, double Ey, double nuYX, double muXY)
	{
		// copied from Julian
		assert(size_ == 2);
		// Note: this isn't the flattened compliance tensor! Rather, it is the
		// matrix inverse of the flattened elasticity tensor.
		stifness_tensor_ << 1.0 / Ex, -nuYX / Ey, 0.0,
			0.0, 1.0 / Ey, 0.0,
			0.0, 0.0, 1.0 / muXY;
	}

	template <int DIM>
	double ElasticityTensor::compute_stress(const std::array<double, DIM> &strain, const int j) const
	{
		double res = 0;

		for (int k = 0; k < DIM; ++k)
			res += (*this)(j, k) * strain[k];

		return res;
	}

	LameParameters::LameParameters()
	{
		lambda_or_E_.emplace_back();
		lambda_or_E_.back().init(1.0);

		mu_or_nu_.emplace_back();
		mu_or_nu_.back().init(1.0);
		size_ = -1;
		is_lambda_mu_ = true;
	}

	void LameParameters::lambda_mu(double px, double py, double pz, double x, double y, double z, int el_id, double &lambda, double &mu) const
	{
		assert(lambda_or_E_.size() == 1 || el_id < lambda_or_E_.size());
		assert(mu_or_nu_.size() == 1 || el_id < mu_or_nu_.size());
		assert(size_ == 2 || size_ == 3);

		const auto &tmp1 = lambda_or_E_.size() == 1 ? lambda_or_E_[0] : lambda_or_E_[el_id];
		const auto &tmp2 = mu_or_nu_.size() == 1 ? mu_or_nu_[0] : mu_or_nu_[el_id];

		double llambda = tmp1(x, y, z, 0, el_id);
		double mmu = tmp2(x, y, z, 0, el_id);

		if (!is_lambda_mu_)
		{
			lambda = convert_to_lambda(size_ == 3, llambda, mmu);
			mu = convert_to_mu(llambda, mmu);
		}
		else
		{
			lambda = llambda;
			mu = mmu;
		}

		assert(!std::isnan(lambda));
		assert(!std::isnan(mu));
		assert(!std::isinf(lambda));
		assert(!std::isinf(mu));
	}

	void LameParameters::add_multimaterial(const int index, const json &params, const bool is_volume)
	{
		const int size = is_volume ? 3 : 2;
		assert(size_ == -1 || size == size_);
		size_ = size;

		for (int i = lambda_or_E_.size(); i <= index; ++i)
		{
			lambda_or_E_.emplace_back();
			mu_or_nu_.emplace_back();
		}

		if (params.count("young"))
		{
			set_e_nu(index, params["young"], params["nu"]);
		}
		else if (params.count("E"))
		{
			set_e_nu(index, params["E"], params["nu"]);
		}
		else if (params.count("lambda"))
		{
			lambda_or_E_[index].init(params["lambda"]);
			mu_or_nu_[index].init(params["mu"]);
			is_lambda_mu_ = true;
		}
	}

	void LameParameters::set_e_nu(const int index, const json &E, const json &nu)
	{
		// TODO: conversion is always called
		is_lambda_mu_ = false;
		lambda_or_E_[index].init(E);
		mu_or_nu_[index].init(nu);
	}

	Density::Density()
	{
		rho_.emplace_back();
		rho_.back().init(1.0);
	}

	double Density::operator()(double px, double py, double pz, double x, double y, double z, int el_id) const
	{
		assert(rho_.size() == 1 || el_id < rho_.size());

		const auto &tmp = rho_.size() == 1 ? rho_[0] : rho_[el_id];
		const double res = tmp(x, y, z, 0, el_id);
		assert(!std::isnan(res));
		assert(!std::isinf(res));
		return res;
	}

	void Density::add_multimaterial(const int index, const json &params)
	{
		for (int i = rho_.size(); i <= index; ++i)
		{
			rho_.emplace_back();
		}

		if (params.count("rho"))
		{
			rho_[index].init(params["rho"]);
		}
		else if (params.count("density"))
		{
			rho_[index].init(params["density"]);
		}
	}

	// template instantiation
	template double ElasticityTensor::compute_stress<3>(const std::array<double, 3> &strain, const int j) const;
	template double ElasticityTensor::compute_stress<6>(const std::array<double, 6> &strain, const int j) const;

} // namespace polyfem

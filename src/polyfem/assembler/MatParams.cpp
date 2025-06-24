#include "MatParams.hpp"

#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <iostream>

namespace polyfem::assembler
{
	namespace
	{
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
	} // namespace

	GenericMatParam::GenericMatParam(const std::string &param_name)
		: param_name_(param_name)
	{
	}

	void GenericMatParam::add_multimaterial(const int index, const json &params, const std::string &unit_type)
	{
		for (int i = param_.size(); i <= index; ++i)
		{
			param_.emplace_back();
		}

		if (params.count(param_name_))
		{
			param_[index].init(params[param_name_]);
			param_[index].set_unit_type(unit_type);
		}
	}

	double GenericMatParam::operator()(const RowVectorNd &p, double t, int index) const
	{
		const double x = p(0);
		const double y = p(1);
		const double z = p.size() == 3 ? p(2) : 0;

		return (*this)(x, y, z, t, index);
	}

	double GenericMatParam::operator()(double x, double y, double z, double t, int index) const
	{
		assert(param_.size() == 1 || index < param_.size());

		const auto &tmp_param = param_.size() == 1 ? param_[0] : param_[index];

		return tmp_param(x, y, z, t, index);
	}

	GenericMatParams::GenericMatParams(const std::string &param_name)
		: param_name_(param_name)
	{
	}

	void GenericMatParams::add_multimaterial(const int index, const json &params, const std::string &unit_type)
	{
		if (!params.contains(param_name_))
			return;

		std::vector<json> params_array = utils::json_as_array(params[param_name_]);
		assert(params_array.size() == params_.size() || params_.empty());

		if (params_.empty())
			for (int i = 0; i < params_array.size(); ++i)
				params_.emplace_back(param_name_ + "_" + std::to_string(i));

		for (int i = 0; i < params_.size(); ++i)
		{
			for (int j = params_.at(i).param_.size(); j <= index; ++j)
			{
				params_.at(i).param_.emplace_back();
			}

			params_.at(i).param_[index].init(params_array[i]);
			params_.at(i).param_[index].set_unit_type(unit_type);
		}
	}

	void ElasticityTensor::resize(const int size)
	{
		if (size == 2)
			stiffness_tensor_.resize(3, 3);
		else
			stiffness_tensor_.resize(6, 6);

		stiffness_tensor_.setZero();

		size_ = size;
	}

	double ElasticityTensor::operator()(int i, int j) const
	{
		if (j < i)
		{
			std::swap(i, j);
		}

		assert(j >= i);
		return stiffness_tensor_(i, j);
	}

	double &ElasticityTensor::operator()(int i, int j)
	{
		if (j < i)
		{
			std::swap(i, j);
		}

		assert(j >= i);
		return stiffness_tensor_(i, j);
	}

	void ElasticityTensor::set_from_entries(const std::vector<double> &entries, const std::string &stress_units)
	{
		if (size_ == 2)
		{
			if (entries.size() == 4)
			{
				set_orthotropic(
					entries[0],
					entries[1],
					entries[2],
					entries[3], stress_units);

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
			if (entries.size() == 5)
			{
				set_transversely_isotropic(
					entries[0],
					entries[1],
					entries[2],
					entries[3],
					entries[4],
					stress_units);

				return;
			}
			else if (entries.size() == 9)
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
					entries[8], stress_units);

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

	void ElasticityTensor::set_from_lambda_mu(const double lambda, const double mu, const std::string &stress_units)
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

	void ElasticityTensor::set_from_young_poisson(const double young, const double nu, const std::string &stress_units)
	{
		if (size_ == 2)
		{
			stiffness_tensor_ << 1.0, nu, 0.0,
				nu, 1.0, 0.0,
				0.0, 0.0, (1.0 - nu) / 2.0;
			stiffness_tensor_ *= young / (1.0 - nu * nu);
		}
		else
		{
			assert(size_ == 3);
			const double v = nu;
			stiffness_tensor_ << 1. - v, v, v, 0, 0, 0,
				v, 1. - v, v, 0, 0, 0,
				v, v, 1. - v, 0, 0, 0,
				0, 0, 0, (1. - 2. * v) / 2., 0, 0,
				0, 0, 0, 0, (1. - 2. * v) / 2., 0,
				0, 0, 0, 0, 0, (1. - 2. * v) / 2.;
			stiffness_tensor_ *= young / ((1. + v) * (1. - 2. * v));
		}
	}

	void ElasticityTensor::set_orthotropic(
		double Ex, double Ey, double Ez,
		double nuXY, double nuXZ, double nuYZ,
		double muYZ, double muZX, double muXY, const std::string &stress_units)
	{
		assert(size_ == 3);

		// from https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_orthotropic.cfm
		double nuYX = nuXY * Ey / Ex;
		double nuZX = nuXZ * Ez / Ex;
		double nuZY = nuYZ * Ez / Ey;

		Eigen::MatrixXd compliance;
		compliance.setZero(6, 6);
		compliance << 1 / Ex, -nuYX / Ey, -nuZX / Ez, 0, 0, 0,
			-nuXY / Ex, 1 / Ey, -nuZY / Ez, 0, 0, 0,
			-nuXZ / Ex, -nuYZ / Ey, 1 / Ez, 0, 0, 0,
			0, 0, 0, 1 / (2 * muYZ), 0, 0,
			0, 0, 0, 0, 1 / (2 * muZX), 0,
			0, 0, 0, 0, 0, 1 / (2 * muXY);
		stiffness_tensor_ = compliance.inverse();
	}

	void ElasticityTensor::set_orthotropic(double Ex, double Ey, double nuXY, double muXY, const std::string &stress_units)
	{
		assert(size_ == 2);

		// from https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_orthotropic.cfm
		double nuYX = nuXY * Ey / Ex;

		Eigen::MatrixXd compliance;
		compliance.setZero(3, 3);
		compliance << 1.0 / Ex, -nuYX / Ey, 0.0,
			-nuXY / Ex, 1.0 / Ey, 0.0,
			0.0, 0.0, 1.0 / (2 * muXY);
		stiffness_tensor_ = compliance.inverse();
	}

	void ElasticityTensor::set_transversely_isotropic(
		double Et, double Ea,
		double nu_t, double nu_a,
		double Ga, const std::string &stress_units)
	{
		assert(size_ == 3);

		// from https://osupdocs.forestry.oregonstate.edu/index.php/Transversely_Isotropic_Material
		Eigen::MatrixXd compliance;
		compliance.setZero(6, 6);
		compliance << 1 / Et, -nu_t / Et, -nu_a / Ea, 0, 0, 0,
			-nu_t / Et, 1 / Et, -nu_a / Ea, 0, 0, 0,
			-nu_a / Ea, -nu_a / Ea, 1 / Ea, 0, 0, 0,
			0, 0, 0, 1 / Ga, 0, 0,
			0, 0, 0, 0, 1 / Ga, 0,
			0, 0, 0, 0, 0, (2 * (1 + nu_t)) / Et;
		stiffness_tensor_ = compliance.inverse();
	}

	template <int DIM>
	double ElasticityTensor::compute_stress(const std::array<double, DIM> &strain, const int j) const
	{
		double res = 0;

		for (int k = 0; k < DIM; ++k)
			res += (*this)(j, k) * strain[k];

		return res;
	}

	void ElasticityTensor::rotate_stiffness(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 6> &rotation_mtx_voigt)
	{
		reference_stiffness_tensor_ = stiffness_tensor_;
		stiffness_tensor_ = rotation_mtx_voigt * stiffness_tensor_ * rotation_mtx_voigt.transpose();
	}

	void ElasticityTensor::unrotate_stiffness()
	{
		stiffness_tensor_ = reference_stiffness_tensor_;
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

	void LameParameters::lambda_mu(double px, double py, double pz, double x, double y, double z, double t, int el_id, double &lambda, double &mu) const
	{
		assert(lambda_or_E_.size() == 1 || el_id < lambda_or_E_.size());
		assert(mu_or_nu_.size() == 1 || el_id < mu_or_nu_.size());
		assert(size_ == 2 || size_ == 3);

		const auto &tmp1 = lambda_or_E_.size() == 1 ? lambda_or_E_[0] : lambda_or_E_[el_id];
		const auto &tmp2 = mu_or_nu_.size() == 1 ? mu_or_nu_[0] : mu_or_nu_[el_id];

		double llambda = tmp1(x, y, z, t, el_id);
		double mmu = tmp2(x, y, z, t, el_id);

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

		if (lambda_mat_.size() > el_id && mu_mat_.size() > el_id)
		{
			lambda = lambda_mat_(el_id);
			mu = mu_mat_(el_id);
		}

		assert(!std::isnan(lambda));
		assert(!std::isnan(mu));
		assert(!std::isinf(lambda));
		assert(!std::isinf(mu));
	}

	void LameParameters::add_multimaterial(const int index, const json &params, const bool is_volume, const std::string &stress_unit)
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
			set_e_nu(index, params["young"], params["nu"], stress_unit);
		}
		else if (params.count("E"))
		{
			set_e_nu(index, params["E"], params["nu"], stress_unit);
		}
		else if (params.count("lambda"))
		{
			lambda_or_E_[index].init(params["lambda"]);
			mu_or_nu_[index].init(params["mu"]);

			lambda_or_E_[index].set_unit_type(stress_unit);
			mu_or_nu_[index].set_unit_type(stress_unit);
			is_lambda_mu_ = true;
		}
	}

	void LameParameters::set_e_nu(const int index, const json &E, const json &nu, const std::string &stress_unit)
	{
		// TODO: conversion is always called
		is_lambda_mu_ = false;
		lambda_or_E_[index].init(E);
		mu_or_nu_[index].init(nu);

		lambda_or_E_[index].set_unit_type(stress_unit);
		// nu has no unit
		mu_or_nu_[index].set_unit_type("");
	}

	Density::Density()
	{
		rho_.emplace_back();
		rho_.back().init(1.0);
	}

	double Density::operator()(double px, double py, double pz, double x, double y, double z, double t, int el_id) const
	{
		assert(rho_.size() == 1 || el_id < rho_.size());

		const auto &tmp = rho_.size() == 1 ? rho_[0] : rho_[el_id];
		const double res = tmp(x, y, z, t, el_id);
		assert(!std::isnan(res));
		assert(!std::isinf(res));
		return res;
	}

	void Density::add_multimaterial(const int index, const json &params, const std::string &density_unit)
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

		rho_[index].set_unit_type(density_unit);
	}

	FiberDirection::FiberDirection()
	{
	}

	void FiberDirection::resize(const int size)
	{
		assert(size == 2 || size == 3);
		size_ = size;
		if (!dir_.empty())
		{
			for (const auto &m : dir_)
			{
				assert(m.rows() == size && m.cols() == size);
			}
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 1, 3, 3> FiberDirection::operator()(double px, double py, double pz, double x, double y, double z, double t, int el_id) const
	{
		assert(dir_.size() == 1 || el_id < dir_.size());

		const auto &tmp = dir_.size() == 1 ? dir_[0] : dir_[el_id];
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 1, 3, 3> res;
		res.resize(tmp.rows(), tmp.cols());
		for (int i = 0; i < tmp.rows(); ++i)
		{
			for (int j = 0; j < tmp.cols(); ++j)
			{
				res(i, j) = tmp(i, j)(x, y, z, t, el_id);

				assert(!std::isnan(res(i, j)));
				assert(!std::isinf(res(i, j)));
			}
		}
		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 1, 6, 6> FiberDirection::stiffness_rotation_voigt(double px, double py, double pz, double x, double y, double z, double t, int el_id) const
	{
		// Rotate stiffness mtx in voigt notation according to:
		// https://scicomp.stackexchange.com/questions/35600/4th-order-tensor-rotation-sources-to-refer

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 1, 3, 3> rot = (*this)(px, py, pz, x, y, z, t, el_id);
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 1, 6, 6> res;

		int dim = rot.rows();

		static const double sqrt2 = std::sqrt(2.0);

		if (dim == 2)
		{
			res.resize(3, 3);
			// Still need to compute for 2d
			assert(false);
			// res << rot(0, 0) * rot(0, 0), rot(0, 1) * rot(0, 1), 0,
			// 	rot(1, 0) * rot(1, 0), rot(1, 1) * rot(1, 1), 0,
			// 	0, 0, 1;
		}
		else
		{
			assert(dim == 3);
			res.resize(6, 6);
			res << rot(0, 0) * rot(0, 0), rot(0, 1) * rot(0, 1), rot(0, 2) * rot(0, 2), sqrt2 * rot(0, 1) * rot(0, 2), sqrt2 * rot(0, 0) * rot(0, 2), sqrt2 * rot(0, 0) * rot(0, 1),
				rot(1, 0) * rot(1, 0), rot(1, 1) * rot(1, 1), rot(1, 2) * rot(1, 2), sqrt2 * rot(1, 1) * rot(1, 2), sqrt2 * rot(1, 0) * rot(1, 2), sqrt2 * rot(1, 0) * rot(1, 1),
				rot(2, 0) * rot(2, 0), rot(2, 1) * rot(2, 1), rot(2, 2) * rot(2, 2), sqrt2 * rot(2, 1) * rot(2, 2), sqrt2 * rot(2, 0) * rot(2, 2), sqrt2 * rot(2, 0) * rot(2, 1),
				sqrt2 * rot(1, 0) * rot(2, 0), sqrt2 * rot(1, 1) * rot(2, 1), sqrt2 * rot(1, 2) * rot(2, 2), rot(1, 1) * rot(2, 2) + rot(1, 2) * rot(2, 1), rot(1, 0) * rot(2, 2) + rot(1, 2) * rot(2, 0), rot(1, 0) * rot(2, 1) + rot(1, 1) * rot(2, 0),
				sqrt2 * rot(0, 0) * rot(2, 0), sqrt2 * rot(0, 1) * rot(2, 1), sqrt2 * rot(0, 2) * rot(2, 2), rot(0, 1) * rot(2, 2) + rot(0, 2) * rot(2, 1), rot(0, 0) * rot(2, 2) + rot(0, 2) * rot(2, 0), rot(0, 0) * rot(2, 1) + rot(0, 1) * rot(2, 0),
				sqrt2 * rot(0, 0) * rot(1, 0), sqrt2 * rot(0, 1) * rot(1, 1), sqrt2 * rot(0, 2) * rot(1, 2), rot(0, 1) * rot(1, 2) + rot(0, 2) * rot(1, 1), rot(0, 0) * rot(1, 2) + rot(0, 2) * rot(1, 0), rot(0, 0) * rot(1, 1) + rot(0, 1) * rot(1, 0);
		}

		return res;
	}

	void FiberDirection::add_multimaterial(const int index, const json &dir, const std::string &unit)
	{
		for (int i = dir_.size(); i <= index; ++i)
		{
			dir_.emplace_back();
		}

		if (dir.size() == 3 || dir.size() == 2)
		{
			const int size = dir.size();
			assert(size == size_);
			dir_[index].resize(size, size);
			for (int i = 0; i < size; ++i)
			{
				if (!dir[i].is_array() || dir[i].size() != size)
				{
					log_and_throw_error(fmt::format("Fiber must be {}x{}, row {} is {}", size, size, i, dir[i].dump()));
				}
				for (int j = 0; j < size; ++j)
				{
					dir_[index](i, j).init(dir[i][j]);
					dir_[index](i, j).set_unit_type(unit);
				}
			}
			has_rotation_ = true;
		}
		else if (dir.size() == 9 || dir.size() == 4)
		{
			const int size = dir.size() == 9 ? 3 : 2;
			assert(size == size_);
			dir_[index].resize(size, size);
			for (int i = 0; i < size; ++i)
			{
				for (int j = 0; j < size; ++j)
				{
					dir_[index](i, j).init(dir[i * size + j]);
					dir_[index](i, j).set_unit_type(unit);
				}
			}
			has_rotation_ = true;
		}
		else if (dir.empty())
		{
			dir_[index].resize(size_, size_);
			for (int i = 0; i < size_; ++i)
			{
				for (int j = 0; j < size_; ++j)
				{
					dir_[index](i, j).init(i == j ? 1.0 : 0.0);
					dir_[index](i, j).set_unit_type(unit);
				}
			}
			has_rotation_ = false;
		}
		else
		{
			log_and_throw_error("Fiber direction must be a 3x3 or 2x2 matrix");
		}
	}

	// template instantiation
	template double ElasticityTensor::compute_stress<3>(const std::array<double, 3> &strain, const int j) const;
	template double ElasticityTensor::compute_stress<6>(const std::array<double, 6> &strain, const int j) const;

} // namespace polyfem::assembler
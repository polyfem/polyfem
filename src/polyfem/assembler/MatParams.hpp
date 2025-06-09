#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/ExpressionValue.hpp>

namespace polyfem::assembler
{
	class GenericMatParam
	{
	public:
		GenericMatParam(const std::string &param_name);

		double operator()(const RowVectorNd &p, double t, int index) const;
		double operator()(double x, double y, double z, double t, int index) const;

		void add_multimaterial(const int index, const json &params, const std::string &unit_type);

	private:
		const std::string param_name_;
		std::vector<utils::ExpressionValue> param_;

		friend class GenericMatParams;
	};

	class GenericMatParams
	{
	public:
		GenericMatParams(const std::string &param_name);

		const GenericMatParam &operator[](const size_t i) const { return params_[i]; }
		size_t size() const { return params_.size(); }

		void add_multimaterial(const int index, const json &params, const std::string &unit_type);

	private:
		const std::string param_name_;
		std::vector<GenericMatParam> params_;
	};

	class ElasticityTensor
	{
	public:
		void resize(const int size);

		double operator()(int i, int j) const;
		double &operator()(int i, int j);

		void set_from_entries(const std::vector<double> &entries, const std::string &stress_unit);
		void set_from_lambda_mu(const double lambda, const double mu, const std::string &stress_unit);
		void set_from_young_poisson(const double young, const double poisson, const std::string &stress_unit);

		void set_orthotropic(
			double Ex, double Ey, double Ez,
			double nuXY, double nuXZ, double nuYZ,
			double muYZ, double muZX, double muXY, const std::string &stress_unit);
		void set_orthotropic(double Ex, double Ey, double nuXY, double muXY, const std::string &stress_unit);
		void set_transversely_isotropic(
			double Et, double Ea,
			double nu_t, double nu_a,
			double Ga, const std::string &stress_units);

		template <int DIM>
		double compute_stress(const std::array<double, DIM> &strain, const int j) const;

		void rotate_stiffness(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 6> &rotation_mtx_voigt);
		void unrotate_stiffness();

	private:
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 6> stiffness_tensor_;
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 6> reference_stiffness_tensor_;
		int size_;
	};

	class LameParameters
	{
	public:
		LameParameters();

		void add_multimaterial(const int index, const json &params, const bool is_volume, const std::string &stress_unit);

		void lambda_mu(double px, double py, double pz, double x, double y, double z, double t, int el_id, double &lambda, double &mu) const;
		void lambda_mu(const Eigen::MatrixXd &param, const Eigen::MatrixXd &p, double t, int el_id, double &lambda, double &mu) const
		{
			assert(param.size() == 2 || param.size() == 3);
			assert(param.size() == p.size());
			lambda_mu(
				param(0), param(1), param.size() == 3 ? param(2) : 0.0,
				p(0), p(1), p.size() == 3 ? p(2) : 0.0,
				t,
				el_id, lambda, mu);
		}

		Eigen::MatrixXd lambda_mat_, mu_mat_;

	private:
		void set_e_nu(const int index, const json &E, const json &nu, const std::string &stress_unit);

		int size_;
		std::vector<utils::ExpressionValue> lambda_or_E_, mu_or_nu_;
		bool is_lambda_mu_;
	};

	class Density
	{
	public:
		Density();
		virtual ~Density() = default;

		virtual void add_multimaterial(const int index, const json &params, const std::string &density_unit);

		virtual double operator()(double px, double py, double pz, double x, double y, double z, double t, int el_id) const;
		double operator()(const Eigen::MatrixXd &param, const Eigen::MatrixXd &p, double t, int el_id) const
		{
			assert(param.size() == 2 || param.size() == 3);
			assert(param.size() == p.size());
			return (*this)(param(0), param(1), param.size() == 3 ? param(2) : 0.0,
						   p(0), p(1), p.size() == 3 ? p(2) : 0.0,
						   t, el_id);
		}

	private:
		void set_rho(const json &rho);

		std::vector<utils::ExpressionValue> rho_;
	};

	class NoDensity : public Density
	{
	public:
		NoDensity() {}

		void add_multimaterial(const int index, const json &params, const std::string &density_unit) override
		{
			throw std::runtime_error("NoDensity does not support multimaterial");
		}

		double operator()(double px, double py, double pz, double x, double y, double z, double t, int el_id) const override
		{
			return 1.0;
		}
	};

	class FiberDirection
	{
	public:
		FiberDirection();
		virtual ~FiberDirection() = default;

		void resize(const int size);

		void add_multimaterial(const int index, const json &params, const std::string &unit);

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 1, 3, 3> operator()(double px, double py, double pz, double x, double y, double z, double t, int el_id) const;

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 1, 3, 3> operator()(const Eigen::MatrixXd &param, const Eigen::MatrixXd &p, double t, int el_id) const
		{
			assert(param.size() == 2 || param.size() == 3);
			assert(param.size() == p.size());
			return (*this)(param(0), param(1), param.size() == 3 ? param(2) : 0.0,
						   p(0), p(1), p.size() == 3 ? p(2) : 0.0,
						   t, el_id);
		}

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 1, 6, 6> stiffness_rotation_voigt(double px, double py, double pz, double x, double y, double z, double t, int el_id) const;

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 1, 6, 6> stiffness_rotation_voigt(const Eigen::MatrixXd &param, const Eigen::MatrixXd &p, double t, int el_id) const
		{
			assert(param.size() == 2 || param.size() == 3);
			assert(param.size() == p.size());
			return this->stiffness_rotation_voigt(param(0), param(1), param.size() == 3 ? param(2) : 0.0,
												  p(0), p(1), p.size() == 3 ? p(2) : 0.0,
												  t, el_id);
		}

		bool has_rotation() const { return has_rotation_; }

	private:
		std::vector<Eigen::Matrix<utils::ExpressionValue, Eigen::Dynamic, Eigen::Dynamic, 1, 3, 3>> dir_;
		int size_;
		bool has_rotation_;
	};

} // namespace polyfem::assembler
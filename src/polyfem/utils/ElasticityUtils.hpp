#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>

#include <vector>
#include <array>
#include <functional>

namespace polyfem
{
	constexpr int SMALL_N = POLYFEM_SMALL_N;
	constexpr int BIG_N = POLYFEM_BIG_N;

	Eigen::VectorXd gradient_from_energy(const int size, const int n_bases, const assembler::ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 6, 1>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun6,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 8, 1>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun8,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 12, 1>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun12,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 18, 1>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun18,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 24, 1>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun24,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 30, 1>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun30,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 60, 1>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun60,
										 const std::function<DScalar1<double, Eigen::Matrix<double, 81, 1>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun81,
										 const std::function<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funN,
										 const std::function<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 1000, 1>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funBigN,
										 const std::function<DScalar1<double, Eigen::VectorXd>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funn);

	Eigen::MatrixXd hessian_from_energy(const int size, const int n_bases, const assembler::ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da,
										const std::function<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun6,
										const std::function<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun8,
										const std::function<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun12,
										const std::function<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun18,
										const std::function<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun24,
										const std::function<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun30,
										const std::function<DScalar2<double, Eigen::Matrix<double, 60, 1>, Eigen::Matrix<double, 60, 60>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun60,
										const std::function<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun81,
										const std::function<DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, SMALL_N, SMALL_N>>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funN,
										const std::function<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>(const assembler::ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funn);

	double von_mises_stress_for_stress_tensor(const Eigen::MatrixXd &stress);
	void compute_diplacement_grad(const int size, const basis::ElementBases &bs, const assembler::ElementAssemblyValues &vals, const Eigen::MatrixXd &local_pts, const int p, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &displacement_grad);

	double convert_to_lambda(const bool is_volume, const double E, const double nu);
	double convert_to_mu(const double E, const double nu);

	class ElasticityTensor
	{
	public:
		void resize(const int size);

		double operator()(int i, int j) const;
		double &operator()(int i, int j);

		void set_from_entries(const std::vector<double> &entries);
		void set_from_lambda_mu(const double lambda, const double mu);
		void set_from_young_poisson(const double young, const double poisson);

		void set_orthotropic(
			double Ex, double Ey, double Ez,
			double nuYX, double nuZX, double nuZY,
			double muYZ, double muZX, double muXY);
		void set_orthotropic(double Ex, double Ey, double nuYX, double muXY);

		template <int DIM>
		double compute_stress(const std::array<double, DIM> &strain, const int j) const;

	private:
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 6> stifness_tensor_;
		int size_;
	};

	class LameParameters
	{
	public:
		LameParameters();

		void add_multimaterial(const int index, const json &params, const bool is_volume);

		void lambda_mu(double px, double py, double pz, double x, double y, double z, int el_id, double &lambda, double &mu) const;
		void lambda_mu(const Eigen::MatrixXd &param, const Eigen::MatrixXd &p, int el_id, double &lambda, double &mu) const
		{
			assert(param.size() == p.size());
			lambda_mu(
				param(0), param(1), param.size() == 2 ? 0. : param(2),
				p(0), p(1), p.size() ? 0. : p(2),
				el_id, lambda, mu);
		}

	private:
		void set_e_nu(const int index, const json &E, const json &nu);

		int size_;
		std::vector<utils::ExpressionValue> lambda_or_E_, mu_or_nu_;
		bool is_lambda_mu_;
	};

	class Density
	{
	public:
		Density();

		void add_multimaterial(const int index, const json &params);

		double operator()(double px, double py, double pz, double x, double y, double z, int el_id) const;
		double operator()(const Eigen::MatrixXd &param, const Eigen::MatrixXd &p, int el_id) const
		{
			assert(param.size() == p.size());
			return (*this)(param(0), param(1), param.size() == 2 ? 0. : param(2),
						   p(0), p(1), p.size() ? 0. : p(2),
						   el_id);
		}

	private:
		void set_rho(const json &rho);

		std::vector<utils::ExpressionValue> rho_;
	};
} // namespace polyfem

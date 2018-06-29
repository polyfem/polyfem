#pragma once


#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElementBases.hpp>
#include <polyfem/AutodiffTypes.hpp>
#include <polyfem/Types.hpp>

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <functional>


namespace polyfem
{
	Eigen::VectorXd gradient_from_energy(const int size, const int n_bases, const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da,
		const std::function<DScalar1<double, Eigen::Matrix<double, 6, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun6,
		const std::function<DScalar1<double, Eigen::Matrix<double, 8, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun8,
		const std::function<DScalar1<double, Eigen::Matrix<double, 12, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun12,
		const std::function<DScalar1<double, Eigen::Matrix<double, 18, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun18,
		const std::function<DScalar1<double, Eigen::Matrix<double, 24, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun24,
		const std::function<DScalar1<double, Eigen::Matrix<double, 30, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun30,
		const std::function<DScalar1<double, Eigen::Matrix<double, 60, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun60,
		const std::function<DScalar1<double, Eigen::Matrix<double, 81, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun81,
		const std::function<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 90, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funN,
		const std::function<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 1000, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funBigN,
		const std::function<DScalar1<double, Eigen::VectorXd>				(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funn
		);


	Eigen::MatrixXd hessian_from_energy(const int size, const int n_bases, const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const QuadratureVector &da,
		const std::function<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>		(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun6,
		const std::function<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>		(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun8,
		const std::function<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun12,
		const std::function<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun18,
		const std::function<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun24,
		const std::function<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun30,
		const std::function<DScalar2<double, Eigen::Matrix<double, 60, 1>, Eigen::Matrix<double, 60, 60>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun60,
		const std::function<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &fun81,
		const std::function<DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 90, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 90, 90>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funN,
		const std::function<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>								(const ElementAssemblyValues &, const Eigen::MatrixXd &, const QuadratureVector &)> &funn
		);


	double von_mises_stress_for_stress_tensor(const Eigen::MatrixXd &stress);


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
		double   Ex, double   Ey, double   Ez,
		double nuYX, double nuZX, double nuZY,
		double muYZ, double muZX, double muXY);
		void set_orthotropic(double Ex, double Ey, double nuYX, double muXY);

		template<int DIM>
		double compute_stress(const std::array<double, DIM> &strain, const int j) const;

	private:
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 6> stifness_tensor_;
		int size_;
	};
}

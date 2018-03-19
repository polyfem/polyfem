#pragma once


#include "ElementAssemblyValues.hpp"
#include "ElementBases.hpp"
#include "AutodiffTypes.hpp"
#include "Types.hpp"

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <functional>


namespace poly_fem
{
	Eigen::VectorXd gradient_from_energy(const int size, const int n_bases, const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da,
		const std::function<DScalar1<double, Eigen::Matrix<double, 6, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun6,
		const std::function<DScalar1<double, Eigen::Matrix<double, 8, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun8,
		const std::function<DScalar1<double, Eigen::Matrix<double, 12, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun12,
		const std::function<DScalar1<double, Eigen::Matrix<double, 18, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun18,
		const std::function<DScalar1<double, Eigen::Matrix<double, 24, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun24,
		const std::function<DScalar1<double, Eigen::Matrix<double, 30, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun30,
		const std::function<DScalar1<double, Eigen::Matrix<double, 81, 1>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun81,
		const std::function<DScalar1<double, Eigen::VectorXd>				(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &funn
		);


	Eigen::MatrixXd hessian_from_energy(const int size, const int n_bases, const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da,
		const std::function<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>		(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun6,
		const std::function<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>		(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun8,
		const std::function<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun12,
		const std::function<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun18,
		const std::function<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun24,
		const std::function<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun30,
		// const std::function<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>	(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &fun81,
		const std::function<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>								(const ElementAssemblyValues &, const Eigen::MatrixXd &, const Eigen::VectorXd &)> &funn
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

		template<int DIM>
		double compute_stress(const std::array<double, DIM> &strain, const int j) const;

	private:
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 21, 1> stifness_tensor_;
		int size_;
	};
}
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>

namespace poly_fem
{
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
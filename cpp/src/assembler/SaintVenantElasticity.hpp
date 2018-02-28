#pragma once



#include "ElementAssemblyValues.hpp"
#include "ElementBases.hpp"

#include <Eigen/Dense>
#include <array>

namespace poly_fem
{
	class SaintVenantElasticity
	{
	public:
		SaintVenantElasticity();

		// res is R^{dim}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		assemble(const ElementAssemblyValues &vals, const int j, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const;

		// res is R^{n_bases*dim x dim}
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, Eigen::Dynamic, 3>
		assemble_grad(const ElementAssemblyValues &vals, const int j, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const;

		double compute_energy(const ElementAssemblyValues &vals, const Eigen::MatrixXd &displacement, const Eigen::VectorXd &da) const;


		inline int size() const { return size_; }
		void set_size(const int size);

		void set_stiffness_tensor(int i, int j, const double val);
		double stifness_tensor(int i, int j) const;
		void set_lambda_mu(const double lambda, const double mu);

		void compute_von_mises_stresses(const ElementBases &bs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;


	private:
		int size_ = 2;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 21, 1> stifness_tensor_;

		template <typename T, unsigned long N>
		T stress(const std::array<T, N> &strain, const int j) const;

		template <typename T>
		Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1>
		assemble_aux(const ElementAssemblyValues &vals, const int j,
			const Eigen::VectorXd &da, const Eigen::Matrix<T, Eigen::Dynamic, 1> &local_disp) const;
	};
}


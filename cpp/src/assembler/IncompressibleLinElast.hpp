#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElementBases.hpp>

#include <polyfem/AutodiffTypes.hpp>

#include <Eigen/Dense>
#include <functional>


namespace polyfem
{
	class IncompressibleLinearElasticityDispacement
	{
	public:
		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void set_size(const int size);
		inline int size() const { return size_; }

		inline double &mu() { return mu_; }
		inline double mu() const { return mu_; }

		inline double &lambda() { return lambda_; }
		inline double lambda() const { return lambda_; }

		void set_parameters(const json &params);

		void compute_von_mises_stresses(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &stresses) const;
		void compute_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, Eigen::MatrixXd &tensor) const;
	private:
		int size_ = 2;
		double mu_ = 1;
		double lambda_ = 1;

		void assign_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &displacement, const int all_size, Eigen::MatrixXd &all, const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const;
	};

	class IncompressibleLinearElasticityMixed
	{
	public:
		// res is R^{dim}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		assemble(const ElementAssemblyValues &psi_vals, const ElementAssemblyValues &phi_vals, const int i, const int j, const QuadratureVector &da) const;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void set_size(const int size);

		inline double &mu() { return mu_; }
		inline double mu() const { return mu_; }

		inline double &lambda() { return lambda_; }
		inline double lambda() const { return lambda_; }

		inline int rows() const { return size_; }
		inline int cols() const { return 1; }

		void set_parameters(const json &params);
	private:
		int size_ = 2;
		double mu_ = 1;
		double lambda_ = 1;
	};


	class IncompressibleLinearElasticityPressure
	{
	public:
		// res is R^{1}
		Eigen::Matrix<double, 1, 1>
		assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const;

		Eigen::Matrix<double, 1, 1>
		compute_rhs(const AutodiffHessianPt &pt) const
		{
			assert(false);
			return Eigen::Matrix<double, 1, 1>::Zero(1,1);
		}

		inline int size() const { return 1; }

		void set_parameters(const json &params);

				inline double &mu() { return mu_; }
		inline double mu() const { return mu_; }

		inline double &lambda() { return lambda_; }
		inline double lambda() const { return lambda_; }

	private:
		double mu_ = 1;
		double lambda_ = 1;
	};
}

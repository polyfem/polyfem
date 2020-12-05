#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/ElementAssemblyValues.hpp>
#include <polyfem/ElementBases.hpp>

#include <polyfem/AutodiffTypes.hpp>

#include <Eigen/Dense>
#include <functional>

namespace polyfem
{
	//stokes local assembler for velocity
	class StokesVelocity
	{
	public:
		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const;

		//not implemented!
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void set_size(const int size);
		inline int size() const { return size_; }

		void set_parameters(const json &params);

		void compute_norm_velocity(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &norms) const;
		void compute_stress_tensor(const ElementBases &bs, const ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &tensor) const;

	private:
		int size_ = 2;
		double viscosity_ = 1;
	};

	//stokes mixed assembler (velocity phi and pressure psi)
	class StokesMixed
	{
	public:
		// res is R^{dim}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		assemble(const ElementAssemblyValues &psi_vals, const ElementAssemblyValues &phi_vals, const int i, const int j, const QuadratureVector &da) const;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void set_size(const int size);

		inline int rows() const { return size_; }
		inline int cols() const { return 1; }

		void set_parameters(const json &params);

	private:
		int size_ = 2;
	};

	//pressure only for stokes is zero
	class StokesPressure
	{
	public:
		// res is R^{dim²}
		Eigen::Matrix<double, 1, 1>
		assemble(const ElementAssemblyValues &vals, const int i, const int j, const QuadratureVector &da) const
		{
			return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
		}

		Eigen::Matrix<double, 1, 1>
		compute_rhs(const AutodiffHessianPt &pt) const
		{
			assert(false);
			return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
		}

		inline int size() const { return 1; }

		void set_parameters(const json &params) {}
	};
} // namespace polyfem

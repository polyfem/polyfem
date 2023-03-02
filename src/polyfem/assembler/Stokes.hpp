#pragma once

#include "AssemblerData.hpp"

#include <polyfem/Common.hpp>

#include <polyfem/utils/AutodiffTypes.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <functional>

namespace polyfem::assembler
{
	// stokes local assembler for velocity
	class StokesVelocity
	{
	public:
		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const;

		// not implemented!
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void set_size(const int size);
		inline int size() const { return size_; }

		void add_multimaterial(const int index, const json &params);

		void compute_norm_velocity(const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &norms) const;
		void compute_stress_tensor(const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &tensor) const;

		double viscosity() const { return viscosity_; }

	private:
		int size_ = 2;
		double viscosity_ = 1;
	};

	// stokes mixed assembler (velocity phi and pressure psi)
	class StokesMixed
	{
	public:
		// res is R^{dim}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		assemble(const MixedAssemblerData &data) const;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void set_size(const int size);

		inline int rows() const { return size_; }
		inline int cols() const { return 1; }

		void add_multimaterial(const int index, const json &params) {}

	private:
		int size_ = 2;
	};

	// pressure only for stokes is zero
	class StokesPressure
	{
	public:
		// res is R^{dim²}
		Eigen::Matrix<double, 1, 1>
		assemble(const LinearAssemblerData &data) const
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

		void add_multimaterial(const int index, const json &params) {}
	};
} // namespace polyfem::assembler

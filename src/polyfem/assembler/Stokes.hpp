#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

namespace polyfem::assembler
{
	// stokes local assembler for velocity
	class StokesVelocity : public TensorLinearAssembler
	{
	public:
		using TensorLinearAssembler::assemble;

		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override;

		// not implemented!
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void add_multimaterial(const int index, const json &params);

		void compute_norm_velocity(const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &norms) const;
		void compute_stress_tensor(const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &tensor) const;

		double viscosity() const { return viscosity_; }

	private:
		double viscosity_ = 1;
	};

	// stokes mixed assembler (velocity phi and pressure psi)
	class StokesMixed : public TensorMixedAssembler
	{
	public:
		using TensorMixedAssembler::assemble;

		// res is R^{dim}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		assemble(const MixedAssemblerData &data) const override;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		inline int rows() const override { return size(); }
		inline int cols() const override { return 1; }

		void add_multimaterial(const int index, const json &params) {}
	};

	// pressure only for stokes is zero
	class StokesPressure : public ScalarLinearAssembler
	{
	public:
		using ScalarLinearAssembler::assemble;

		// res is R^{dim²}
		Eigen::Matrix<double, 1, 1>
		assemble(const LinearAssemblerData &data) const override
		{
			return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
		}

		Eigen::Matrix<double, 1, 1>
		compute_rhs(const AutodiffHessianPt &pt) const
		{
			assert(false);
			return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
		}

		void add_multimaterial(const int index, const json &params) {}
	};
} // namespace polyfem::assembler

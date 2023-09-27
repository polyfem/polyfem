#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

namespace polyfem::assembler
{
	// stokes local assembler for velocity
	class StokesVelocity : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		VectorNd compute_rhs(const AutodiffHessianPt &pt) const override;
		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override;

		void add_multimaterial(const int index, const json &params, const Units &units) override;

		double viscosity() const { return viscosity_; }

		virtual std::string name() const override { return "Stokes"; }
		std::map<std::string, ParamFunc> parameters() const override;

		bool is_fluid() const override { return true; }

	private:
		double viscosity_ = 1;
	};

	// stokes mixed assembler (velocity phi and pressure psi)
	class StokesMixed : public MixedAssembler
	{
	public:
		std::string name() const override { return "StokesMixed"; }

		// res is R^{dim}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		assemble(const MixedAssemblerData &data) const override;

		inline int rows() const override { return size(); }
		inline int cols() const override { return 1; }
	};

	// pressure only for stokes is zero
	class StokesPressure : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		std::string name() const override { return "StokesPressure"; }
		std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }

		// res is R^{dim²}
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override
		{
			return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
		}

		bool is_fluid() const override { return true; }

		void set_size(const int) override { size_ = 1; }
	};

	class OperatorSplitting : public StokesVelocity
	{
	public:
		OperatorSplitting() = default;
		~OperatorSplitting() = default;
		
		std::string name() const override { return "OperatorSplitting"; }
	};
} // namespace polyfem::assembler

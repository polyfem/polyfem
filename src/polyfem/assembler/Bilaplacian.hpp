#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

namespace polyfem::assembler
{
	// local assembler for bilaplacian, see laplacian
	//  0 L
	//  L M
	class BilaplacianMain : public ScalarLinearAssembler
	{
	public:
		using ScalarLinearAssembler::assemble;

		Eigen::Matrix<double, 1, 1>
		assemble(const LinearAssemblerData &data) const override
		{
			return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
		}

		Eigen::Matrix<double, 1, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void add_multimaterial(const int index, const json &params) {}
	};

	class BilaplacianMixed : public ScalarMixedAssembler
	{
	public:
		using ScalarMixedAssembler::assemble;

		// res is R
		Eigen::Matrix<double, 1, 1>
		assemble(const MixedAssemblerData &data) const override;

		Eigen::Matrix<double, 1, 1>
		compute_rhs(const AutodiffHessianPt &pt) const
		{
			assert(false);
			return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
		}

		inline int rows() const override { return 1; }
		inline int cols() const override { return 1; }

		void add_multimaterial(const int index, const json &params) {}
	};

	class BilaplacianAux : public ScalarLinearAssembler
	{
	public:
		using ScalarLinearAssembler::assemble;

		// res is R
		Eigen::Matrix<double, 1, 1>
		assemble(const LinearAssemblerData &data) const override;

		Eigen::Matrix<double, 1, 1>
		compute_rhs(const AutodiffHessianPt &pt) const
		{
			assert(false);
			return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
		}

		void add_multimaterial(const int index, const json &params) {}
	};
} // namespace polyfem::assembler

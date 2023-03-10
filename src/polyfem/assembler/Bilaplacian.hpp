#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

namespace polyfem::assembler
{
	// local assembler for bilaplacian, see laplacian
	//  0 L
	//  L M
	class BilaplacianMain : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override
		{
			return Eigen::Matrix<double, 1, 1>::Zero(1, 1);
		}

		std::string name() const override { return "BiLaplacian"; }
		std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }
	};

	class BilaplacianMixed : public MixedAssembler
	{
	public:
		std::string name() const override { return "BilaplacianMixed"; }

		using MixedAssembler::assemble;

		// res is R
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		assemble(const MixedAssemblerData &data) const override;

		inline int rows() const override { return 1; }
		inline int cols() const override { return 1; }
	};

	class BilaplacianAux : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		// res is R
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override;

		std::string name() const override { return "BiLaplacianAux"; }
		std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }
	};
} // namespace polyfem::assembler

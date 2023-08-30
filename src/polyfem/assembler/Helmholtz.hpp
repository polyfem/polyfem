#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

// local assembler for helmolhz equation, see Laplace
namespace polyfem::assembler
{
	class Helmholtz : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override;
		VectorNd compute_rhs(const AutodiffHessianPt &pt) const override;

		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> kernel(const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const override;

		// sets the k parameter
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		double k() const { return k_; }

		std::string name() const override { return "Helmholtz"; }
		std::map<std::string, ParamFunc> parameters() const override;

	private:
		double k_ = 1;
	};
} // namespace polyfem::assembler

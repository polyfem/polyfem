#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

// Navier-Stokes local assembler
namespace polyfem::assembler
{
	class NavierStokesVelocity : public NLAssembler
	{
	public:
		using NLAssembler::assemble_energy;
		using NLAssembler::assemble_gradient;
		using NLAssembler::assemble_hessian;

		std::string name() const override { return "NavierStokes"; }
		std::map<std::string, ParamFunc> parameters() const override;

		// navier stokes is not energy based
		double compute_energy(const NonLinearAssemblerData &data) const override
		{
			// not used, this formulation is gradient based!
			assert(false);
			return 0;
		}

		// res is R^{dim²}
		// pde
		Eigen::VectorXd
		assemble_gradient(const NonLinearAssemblerData &data) const override;

		Eigen::MatrixXd
		// gradient of pde, this returns full gradient or partil depending on the template
		assemble_hessian(const NonLinearAssemblerData &data) const override;

		// rhs for fabbricated solution
		VectorNd compute_rhs(const AutodiffHessianPt &pt) const override;

		// set viscosity
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		bool is_fluid() const override { return true; }

		void set_picard(const bool val) { full_gradient_ = !val; }

	private:
		double viscosity_ = 1;

		// not full graidnet used for Picard iteration
		bool full_gradient_ = true;

		Eigen::MatrixXd compute_N(const NonLinearAssemblerData &data) const;
		Eigen::MatrixXd compute_W(const NonLinearAssemblerData &data) const;
	};

} // namespace polyfem::assembler

#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

// Navier-Stokes local assembler
namespace polyfem::assembler
{
	template <bool full_gradient>
	// full graidnet used for Picard iteration
	class NavierStokesVelocity : public NLAssembler
	{
	public:
		using NLAssembler::assemble_energy;
		using NLAssembler::assemble_grad;
		using NLAssembler::assemble_hessian;

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
		assemble_grad(const NonLinearAssemblerData &data) const override;

		Eigen::MatrixXd
		// gradient of pde, this returns full gradient or partil depending on the template
		assemble_hessian(const NonLinearAssemblerData &data) const override;

		// rhs for fabbricated solution
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		// set viscosity
		void add_multimaterial(const int index, const json &params);

		// return velociry and norm, for compliancy with elasticity
		void compute_norm_velocity(const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &norms) const;
		void compute_stress_tensor(const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &tensor) const;

	private:
		double viscosity_ = 1;

		Eigen::MatrixXd compute_N(const NonLinearAssemblerData &data) const;
		Eigen::MatrixXd compute_W(const NonLinearAssemblerData &data) const;
	};

} // namespace polyfem::assembler

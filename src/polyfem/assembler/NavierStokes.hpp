#pragma once

#include "Assembler.hpp"
#include "AssemblerData.hpp"

#include <polyfem/Common.hpp>

#include <polyfem/utils/AutodiffTypes.hpp>

#include <Eigen/Dense>
#include <functional>

// Navier-Stokes local assembler
namespace polyfem::assembler
{
	template <bool full_gradient>
	// full graidnet used for Picard iteration
	class NavierStokesVelocity : public NLAssembler
	{
	public:
		// res is R^{dim²}
		// pde
		Eigen::VectorXd
		assemble_grad(const NonLinearAssemblerData &data) const;

		Eigen::MatrixXd
		// gradient of pde, this returns full gradient or partil depending on the template
		assemble_hessian(const NonLinearAssemblerData &data) const;

		// navier stokes is not energy based
		double compute_energy(const NonLinearAssemblerData &data) const
		{
			// not used, this formulation is gradient based!
			assert(false);
			return 0;
		}

		// rhs for fabbricated solution
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		compute_rhs(const AutodiffHessianPt &pt) const;

		void set_size(const int size);
		inline int size() const { return size_; }

		// set viscosity
		void add_multimaterial(const int index, const json &params);

		// return velociry and norm, for compliancy with elasticity
		void compute_norm_velocity(const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &norms) const;
		void compute_stress_tensor(const basis::ElementBases &bs, const basis::ElementBases &gbs, const Eigen::MatrixXd &local_pts, const Eigen::MatrixXd &velocity, Eigen::MatrixXd &tensor) const;

	private:
		int size_ = -1;
		double viscosity_ = 1;

		Eigen::MatrixXd compute_N(const NonLinearAssemblerData &data) const;
		Eigen::MatrixXd compute_W(const NonLinearAssemblerData &data) const;
	};

} // namespace polyfem::assembler

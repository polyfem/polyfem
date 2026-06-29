#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

// local assembler for laplace equation
namespace polyfem
{
	namespace assembler
	{
		class Laplacian : public LinearAssembler, public NLAssembler
		{
		public:
			using LinearAssembler::assemble;
			using NLAssembler::assemble_energy;
			using NLAssembler::assemble_gradient;
			using NLAssembler::assemble_hessian;

			std::string name() const override { return "Laplacian"; }
			std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }

			/// computes local stiffness matrix (1x1) for bases i,j
			/// where i,j is passed in through data
			/// ie integral of grad(phi_i) dot grad(phi_j)
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> assemble(const LinearAssemblerData &data) const override;

			double compute_energy(const NonLinearAssemblerData &data) const override;
			Eigen::VectorXd assemble_gradient(const NonLinearAssemblerData &data) const override;
			Eigen::MatrixXd assemble_hessian(const NonLinearAssemblerData &data) const override;

			/// uses autodiff to compute the rhs for a fabricated solution
			/// in this case it just return pt.getHessian().trace()
			/// pt is the evaluation of the solution at a point
			VectorNd compute_rhs(const AutodiffHessianPt &pt) const override;

			void compute_stress_grad_multiply_mat(const OptAssemblerData &data,
												  const Eigen::MatrixXd &mat,
												  Eigen::MatrixXd &stress,
												  Eigen::MatrixXd &result) const override;

			void compute_stiffness_value(const double t,
										 const assembler::ElementAssemblyValues &vals,
										 const Eigen::MatrixXd &local_pts,
										 const Eigen::MatrixXd &displacement,
										 Eigen::MatrixXd &tensor) const override;

			/// kernel of the pde, used in kernel problem
			Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> kernel(const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const override;

			bool is_linear() const override { return true; }
		};
	} // namespace assembler
} // namespace polyfem

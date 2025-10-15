#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>

// local assembler for laplace equation
namespace polyfem
{
	namespace assembler
	{
		// Laplacian with some extra steps
		class Electrostatics : public LinearAssembler
		{
		public:
			Electrostatics() : epsilon_("epsilon") {}

			using LinearAssembler::assemble;

			std::string name() const override { return "Electrostatics"; }
			std::map<std::string, ParamFunc> parameters() const override;

			// inialize material parameter
			void add_multimaterial(const int index, const json &params, const Units &units) override;

			/// computes local stiffness matrix (1x1) for bases i,j
			/// where i,j is passed in through data
			/// ie integral of grad(phi_i) dot grad(phi_j)
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1> assemble(const LinearAssemblerData &data) const override;

			void compute_stress_grad_multiply_mat(const OptAssemblerData &data,
												  const Eigen::MatrixXd &mat,
												  Eigen::MatrixXd &stress,
												  Eigen::MatrixXd &result) const override;

			void compute_stiffness_value(const double t,
										 const assembler::ElementAssemblyValues &vals,
										 const Eigen::MatrixXd &local_pts,
										 const Eigen::MatrixXd &displacement,
										 Eigen::MatrixXd &tensor) const override;

			double compute_stored_energy(
				const bool is_volume,
				const int n_basis,
				const std::vector<basis::ElementBases> &bases,
				const std::vector<basis::ElementBases> &gbases,
				const AssemblyValsCache &cache,
				const double t,
				const Eigen::MatrixXd &solution);

		private:
			GenericMatParam epsilon_;
		};
	} // namespace assembler
} // namespace polyfem
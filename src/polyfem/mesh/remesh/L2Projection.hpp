#pragma once

#include <polyfem/State.hpp>
#include <polyfem/assembler/RhsAssembler.hpp>
#include <polyfem/assembler/MassMatrixAssembler.hpp>
#include <polyfem/solver/NLProblem.hpp>

namespace polyfem::mesh
{
	using namespace polyfem::solver;
	using namespace polyfem::assembler;
	using namespace polyfem::basis;

	/// @brief Project the quantities in u on to the space spanned by mesh.bases.
	void L2_projection(
		const State &state,
		const RhsAssembler &rhs_assembler,
		const bool is_volume,
		const int size,
		const int n_basis_a,
		const std::vector<ElementBases> &bases_a,
		const std::vector<ElementBases> &gbases_a,
		const int n_basis_b,
		const std::vector<ElementBases> &bases_b,
		const std::vector<ElementBases> &gbases_b,
		const polyfem::assembler::AssemblyValsCache &cache,
		const Eigen::MatrixXd &y,
		Eigen::MatrixXd &x,
		const double t0,
		const double dt,
		const int t,
		const bool lump_mass_matrix = false);

	class L2ProjectionForm : public Form
	{
	public:
		L2ProjectionForm(
			const StiffnessMatrix &M,
			const StiffnessMatrix &A,
			const Eigen::VectorXd &x_prev);

	protected:
		/// @brief Compute the value of the form
		/// @param x Current solution
		/// @return Computed value
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override;

		StiffnessMatrix M_;
		Eigen::VectorXd rhs_;
	};

} // namespace polyfem::mesh
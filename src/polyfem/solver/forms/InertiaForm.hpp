#pragma once

#include "Form.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

namespace polyfem
{
	namespace assembler
	{
		class Mass;
		class AssemblyValsCache;
	} // namespace assembler

	namespace basis
	{
		class ElementBases;
	}
} // namespace polyfem

namespace polyfem::solver
{
	/// @brief Form of the inertia
	class InertiaForm : public Form
	{
	public:
		/// @brief Construct a new Inertia Form object
		/// @param mass Mass matrix
		/// @param time_integrator Time integrator
		InertiaForm(const StiffnessMatrix &mass,
					const time_integrator::ImplicitTimeIntegrator &time_integrator);

		std::string name() const override { return "inertia"; }

		static void force_shape_derivative(
			bool is_volume,
			const int n_geom_bases,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &geom_bases,
			const assembler::Mass &assembler,
			const assembler::AssemblyValsCache &ass_vals_cache,
			const Eigen::MatrixXd &velocity,
			const Eigen::MatrixXd &adjoint,
			Eigen::VectorXd &term);

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
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:
		const StiffnessMatrix &mass_;                                    ///< Mass matrix
		const time_integrator::ImplicitTimeIntegrator &time_integrator_; ///< Time integrator
	};
} // namespace polyfem::solver

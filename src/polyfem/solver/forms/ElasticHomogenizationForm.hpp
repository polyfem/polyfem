#pragma once

#include "Form.hpp"

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include "ElasticForm.hpp"
#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	/// @brief Form of the elasticity potential and forces
	class ElasticHomogenizationForm : public ElasticForm
	{
	public:
		/// @brief Construct a new Elastic Form object
		/// @param state Reference to the simulation state
		ElasticHomogenizationForm(const int n_bases,
					const std::vector<basis::ElementBases> &bases,
					const std::vector<basis::ElementBases> &geom_bases,
					const assembler::AssemblerUtils &assembler,
					const assembler::AssemblyValsCache &ass_vals_cache,
					const std::string &formulation,
					const double dt,
					const bool is_volume): ElasticForm(n_bases, bases, geom_bases, assembler, ass_vals_cache, formulation, dt, is_volume) {}

        void set_macro_field(const Eigen::VectorXd &macro_field) { macro_field_ = macro_field; }

	protected:
		/// @brief Compute the elastic potential value
		/// @param x Current solution
		/// @return Value of the elastic potential
		double value_unweighted(const Eigen::VectorXd &x) const override
        {
            return ElasticForm::value_unweighted(x + macro_field_);
        }

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override
        {
            ElasticForm::first_derivative_unweighted(x + macro_field_, gradv);
        }

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) override
        {
            ElasticForm::second_derivative_unweighted(x + macro_field_, hessian);
        }

	private:
        Eigen::VectorXd macro_field_;
	};
} // namespace polyfem::solver

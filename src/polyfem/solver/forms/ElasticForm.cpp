#include "ElasticForm.hpp"

#include <polyfem/basis/ElementBases.hpp>

#include <polyfem/utils/Timer.hpp>

namespace polyfem::solver
{
	ElasticForm::ElasticForm(const int n_bases,
							 const std::vector<basis::ElementBases> &bases,
							 const std::vector<basis::ElementBases> &geom_bases,
							 const assembler::AssemblerUtils &assembler,
							 const assembler::AssemblyValsCache &ass_vals_cache,
							 const std::string &formulation,
							 const double dt,
							 const bool is_volume)
		: n_bases_(n_bases),
		  bases_(bases),
		  geom_bases_(geom_bases),
		  assembler_(assembler),
		  ass_vals_cache_(ass_vals_cache),
		  formulation_(formulation),
		  dt_(dt),
		  is_volume_(is_volume)
	{
		if (assembler_.is_linear(formulation_))
			compute_cached_stiffness();
	}

	double ElasticForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return assembler_.assemble_energy(
			formulation_, is_volume_, bases_, geom_bases_,
			ass_vals_cache_, dt_, x, x_prev_);
	}

	void ElasticForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		Eigen::MatrixXd grad;
		assembler_.assemble_energy_gradient(
			formulation_, is_volume_, n_bases_, bases_, geom_bases_,
			ass_vals_cache_, dt_, x, x_prev_, grad);
		gradv = grad;
	}

	void ElasticForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("elastic hessian");

		hessian.resize(x.size(), x.size());

		if (assembler_.is_linear(formulation_))
		{
			assert(cached_stiffness_.rows() == x.size() && cached_stiffness_.cols() == x.size());
			hessian = cached_stiffness_;
		}
		else
		{
			// NOTE: mat_cache_ is marked as mutable so we can modify it here
			assembler_.assemble_energy_hessian(
				formulation_, is_volume_, n_bases_, project_to_psd_, bases_,
				geom_bases_, ass_vals_cache_, dt_, x, x_prev_, mat_cache_, hessian);
		}
	}

	bool ElasticForm::is_step_valid(const Eigen::VectorXd &, const Eigen::VectorXd &x1) const
	{
		Eigen::VectorXd grad;
		first_derivative(x1, grad);

		if (grad.array().isNaN().any())
			return false;

		// Check the scalar field in the output does not contain NANs.
		// WARNING: Does not work because the energy is not evaluated at the same quadrature points.
		//          This causes small step lengths in the LS.
		// TVector x1_full;
		// reduced_to_full(x1, x1_full);
		// return state_.check_scalar_value(x1_full, true, false);
		return true;
	}

	void ElasticForm::compute_cached_stiffness()
	{
		if (assembler_.is_linear(formulation_) && cached_stiffness_.size() == 0)
		{
			assembler_.assemble_problem(
				formulation_, is_volume_, n_bases_, bases_, geom_bases_,
				ass_vals_cache_, cached_stiffness_);
		}
	}
} // namespace polyfem::solver

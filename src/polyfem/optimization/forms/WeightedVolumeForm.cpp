#include <polyfem/optimization/forms/WeightedVolumeForm.hpp>

#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>

#include <Eigen/Core>

#include <cassert>

namespace polyfem::solver
{
	double WeightedVolumeForm::value_unweighted_with_param(const Eigen::VectorXd &x) const
	{
		assert(x.size() == varform_->get_mesh().n_elements());

		double val = 0;
		assembler::ElementAssemblyValues vals;
		for (int e = 0; e < varform_->primary_space().basis_list().size(); e++)
		{
			varform_->assembly_cache().compute(e, varform_->get_mesh().is_volume(), varform_->primary_space().basis_list()[e], varform_->primary_space().geometry_basis_list()[e], vals);
			val += (vals.det.array() * vals.quadrature.weights.array()).sum() * x(e);
		}
		return val;
	}

	void WeightedVolumeForm::compute_partial_gradient_with_param(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(x.size() == varform_->get_mesh().n_elements());

		gradv.setZero(x.size());
		assembler::ElementAssemblyValues vals;
		for (int e = 0; e < varform_->primary_space().basis_list().size(); e++)
		{
			varform_->assembly_cache().compute(e, varform_->get_mesh().is_volume(), varform_->primary_space().basis_list()[e], varform_->primary_space().geometry_basis_list()[e], vals);
			gradv(e) = (vals.det.array() * vals.quadrature.weights.array()).sum();
		}
		gradv *= weight();
	}
} // namespace polyfem::solver

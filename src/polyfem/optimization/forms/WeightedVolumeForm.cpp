#include <polyfem/optimization/forms/WeightedVolumeForm.hpp>

#include <polyfem/varforms/VarForm.hpp>

#include <Eigen/Core>

#include <cassert>

namespace polyfem::solver
{
	double WeightedVolumeForm::value_unweighted_with_param(const Eigen::VectorXd &x) const
	{
		assert(x.size() == state_->get_mesh().n_elements());

		double val = 0;
		assembler::ElementAssemblyValues vals;
		for (int e = 0; e < state_->primary_space().basis_list().size(); e++)
		{
			state_->assembly_cache().compute(e, state_->get_mesh().is_volume(), state_->primary_space().basis_list()[e], state_->primary_space().geometry_basis_list()[e], vals);
			val += (vals.det.array() * vals.quadrature.weights.array()).sum() * x(e);
		}
		return val;
	}

	void WeightedVolumeForm::compute_partial_gradient_with_param(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(x.size() == state_->get_mesh().n_elements());

		gradv.setZero(x.size());
		assembler::ElementAssemblyValues vals;
		for (int e = 0; e < state_->primary_space().basis_list().size(); e++)
		{
			state_->assembly_cache().compute(e, state_->get_mesh().is_volume(), state_->primary_space().basis_list()[e], state_->primary_space().geometry_basis_list()[e], vals);
			gradv(e) = (vals.det.array() * vals.quadrature.weights.array()).sum();
		}
		gradv *= weight();
	}
} // namespace polyfem::solver

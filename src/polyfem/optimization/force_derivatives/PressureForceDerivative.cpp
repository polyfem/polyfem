#include "PressureForceDerivative.hpp"

#include <cassert>
#include <Eigen/Core>
#include <polyfem/solver/forms/PressureForm.hpp>
#include <polyfem/utils/Types.hpp>

namespace polyfem::solver
{
	void PressureForceDerivative::force_shape_derivative(
		PressureForm &form,
		const int n_verts,
		const double t,
		const Eigen::MatrixXd &x,
		const Eigen::MatrixXd &adjoint,
		Eigen::VectorXd &term)
	{
		Eigen::MatrixXd adjoint_zeroed = adjoint;
		adjoint_zeroed(form.dirichlet_nodes_, Eigen::all).setZero();

		StiffnessMatrix hessian;
		form.pressure_assembler_.compute_force_jacobian(x, form.local_pressure_boundary_, form.dirichlet_nodes_, form.n_boundary_samples_, t, n_verts, hessian);

		term = -hessian.transpose() * adjoint_zeroed;
	}

	double PressureForceDerivative::force_pressure_derivative(
		PressureForm &form,
		const int n_verts,
		const double t,
		const int pressure_boundary_id,
		const Eigen::MatrixXd &x,
		const Eigen::MatrixXd &adjoint)
	{
		Eigen::MatrixXd adjoint_zeroed = adjoint;
		adjoint_zeroed(form.dirichlet_nodes_, Eigen::all).setZero();

		Eigen::VectorXd pressure_gradv;
		form.pressure_assembler_.compute_grad_volume_id(x, pressure_boundary_id, form.local_pressure_boundary_, form.dirichlet_nodes_, form.n_boundary_samples_, pressure_gradv, t, false);
		pressure_gradv *= -1;

		Eigen::MatrixXd term;
		term = pressure_gradv.transpose() * adjoint_zeroed;
		term *= -1;

		assert(term.size() == 1);
		return term(0);
	}
} // namespace polyfem::solver

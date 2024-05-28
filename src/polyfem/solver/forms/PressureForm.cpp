#include "PressureForm.hpp"

#include <polyfem/io/Evaluator.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/BoundarySampler.hpp>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>

#include <polyfem/solver/AdjointTools.hpp>

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	PressureForm::PressureForm(const int ndof,
							   const std::vector<mesh::LocalBoundary> &local_pressure_boundary,
							   const std::unordered_map<int, std::vector<mesh::LocalBoundary>> &local_pressure_cavity,
							   const std::vector<int> &dirichlet_nodes,
							   const int n_boundary_samples,
							   const assembler::PressureAssembler &pressure_assembler,
							   const bool is_time_dependent)
		: ndof_(ndof),
		  local_pressure_boundary_(local_pressure_boundary),
		  local_pressure_cavity_(local_pressure_cavity),
		  dirichlet_nodes_(dirichlet_nodes),
		  n_boundary_samples_(n_boundary_samples),
		  pressure_assembler_(pressure_assembler)
	{
		t_ = 0;
	}

	double PressureForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		double pressure_energy = pressure_assembler_.compute_energy(x, local_pressure_boundary_, n_boundary_samples_, t_);
		double pressure_cavity_energy = pressure_assembler_.compute_cavity_energy(x, local_pressure_cavity_, n_boundary_samples_, t_);
		return -1 * (pressure_energy + pressure_cavity_energy);
	}

	void PressureForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		// REMEMBER -!!!!!
		Eigen::VectorXd pressure_gradv, pressure_cavity_gradv;
		pressure_assembler_.compute_energy_grad(x, local_pressure_boundary_, dirichlet_nodes_, n_boundary_samples_, t_, pressure_gradv);
		pressure_assembler_.compute_cavity_energy_grad(x, local_pressure_cavity_, dirichlet_nodes_, n_boundary_samples_, t_, pressure_cavity_gradv);
		gradv = pressure_gradv + pressure_cavity_gradv;
		gradv *= -1;
	}

	void PressureForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		StiffnessMatrix pressure_hessian, pressure_cavity_hessian;
		pressure_assembler_.compute_energy_hess(x, local_pressure_boundary_, dirichlet_nodes_, n_boundary_samples_, t_, project_to_psd_, pressure_hessian);
		pressure_assembler_.compute_cavity_energy_hess(x, local_pressure_cavity_, dirichlet_nodes_, n_boundary_samples_, t_, project_to_psd_, pressure_cavity_hessian);
		hessian = pressure_hessian + pressure_cavity_hessian;
		hessian *= -1;
	}

	void PressureForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		this->t_ = t;
	}

	void PressureForm::force_shape_derivative(const int n_verts, const double t, const Eigen::MatrixXd &x, const Eigen::MatrixXd &adjoint, Eigen::VectorXd &term)
	{
		Eigen::MatrixXd adjoint_zeroed = adjoint;
		adjoint_zeroed(dirichlet_nodes_, Eigen::all).setZero();

		StiffnessMatrix hessian;
		pressure_assembler_.compute_force_jacobian(x, local_pressure_boundary_, dirichlet_nodes_, n_boundary_samples_, t, n_verts, hessian);

		term = -hessian.transpose() * adjoint_zeroed;
	}

	double PressureForm::force_pressure_derivative(
		const int n_verts,
		const double t,
		const int pressure_boundary_id,
		const Eigen::MatrixXd &x,
		const Eigen::MatrixXd &adjoint)
	{
		Eigen::MatrixXd adjoint_zeroed = adjoint;
		adjoint_zeroed(dirichlet_nodes_, Eigen::all).setZero();

		Eigen::VectorXd pressure_gradv;
		pressure_assembler_.compute_grad_volume_id(x, pressure_boundary_id, local_pressure_boundary_, dirichlet_nodes_, n_boundary_samples_, pressure_gradv, t, false);
		pressure_gradv *= -1;

		Eigen::MatrixXd term;
		term = pressure_gradv.transpose() * adjoint_zeroed;
		term *= -1;

		assert(term.size() == 1);
		return term(0);
	}

} // namespace polyfem::solver

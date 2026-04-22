#include "PeriodicContactForm.hpp"

#include <polyfem/utils/Types.hpp>
#include <polysolve/nonlinear/PostStepData.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cassert>
#include <vector>

namespace polyfem::solver
{
	PeriodicContactForm::PeriodicContactForm(const ipc::CollisionMesh &periodic_collision_mesh,
											 const Eigen::VectorXi &tiled_to_single,
											 const double dhat,
											 const double avg_mass,
											 const bool use_area_weighting,
											 const bool use_improved_max_operator,
											 const bool use_physical_barrier,
											 const bool use_adaptive_barrier_stiffness,
											 const bool is_time_dependent,
											 const bool enable_shape_derivatives,
											 const ipc::BroadPhaseMethod broad_phase_method,
											 const double ccd_tolerance,
											 const int ccd_max_iterations) : BarrierContactForm(periodic_collision_mesh, dhat, avg_mass, use_area_weighting, use_improved_max_operator, use_physical_barrier, use_adaptive_barrier_stiffness, is_time_dependent, enable_shape_derivatives, broad_phase_method, ccd_tolerance, ccd_max_iterations), tiled_to_single_(tiled_to_single), n_single_dof_(tiled_to_single_.maxCoeff() + 1)
	{
		assert(tiled_to_single_.size() == collision_mesh_.full_num_vertices());

		update_projection();

		// const Eigen::MatrixXd displaced = collision_mesh_.displace_vertices(
		//     Eigen::MatrixXd::Zero(collision_mesh_.full_num_vertices(), collision_mesh_.dim()));

		// io::OBJWriter::write(
		//     "tiled-rest.obj", displaced,
		//     collision_mesh_.edges(), collision_mesh_.faces());
	}

	void PeriodicContactForm::update_projection() const
	{
		const int dim = collision_mesh_.dim();
		const auto &boundary_vertices = collision_mesh_.rest_positions();

		std::vector<Eigen::Triplet<double>> entries;
		for (int i = 0; i < collision_mesh_.num_vertices(); i++)
		{
			const int i_full = collision_mesh_.to_full_vertex_id(i);
			for (int d = 0; d < dim; d++)
				entries.emplace_back(tiled_to_single_(i_full) * dim + d, i_full * dim + d, 1);

			for (int p = 0; p < dim; p++)
				for (int q = 0; q < dim; q++)
					entries.emplace_back(n_single_dof_ * dim + p * dim + q, i_full * dim + p, boundary_vertices(i, q));
		}

		proj.resize(n_single_dof_ * dim + dim * dim, tiled_to_single_.size() * dim);
		proj.setZero();
		proj.setFromTriplets(entries.begin(), entries.end());
	}

	Eigen::VectorXd PeriodicContactForm::single_to_tiled(const Eigen::VectorXd &x) const
	{
		const int dim = collision_mesh_.dim();
		const auto &boundary_vertices = collision_mesh_.rest_positions();
		assert(x.size() == n_single_dof_ * dim + dim * dim);

		Eigen::VectorXd tiled_x;
		tiled_x.setZero(tiled_to_single_.size() * dim);
		for (int i = 0; i < collision_mesh_.num_vertices(); i++)
		{
			const int i_full = collision_mesh_.to_full_vertex_id(i);
			for (int d = 0; d < dim; d++)
				tiled_x(i_full * dim + d) += x(tiled_to_single_(i_full) * dim + d);

			for (int p = 0; p < dim; p++)
				for (int q = 0; q < dim; q++)
					tiled_x(i_full * dim + p) += x(n_single_dof_ * dim + p * dim + q) * boundary_vertices(i, q);
		}

		return tiled_x;
	}
	Eigen::VectorXd PeriodicContactForm::tiled_to_single_grad(const Eigen::VectorXd &grad) const
	{
		const int dim = collision_mesh_.dim();
		const auto &boundary_vertices = collision_mesh_.rest_positions();
		assert(grad.size() == tiled_to_single_.size() * dim);

		Eigen::VectorXd reduced_grad;
		reduced_grad.setZero(n_single_dof_ * dim + dim * dim);
		for (int i = 0; i < collision_mesh_.num_vertices(); i++)
		{
			const int i_full = collision_mesh_.to_full_vertex_id(i);
			for (int d = 0; d < dim; d++)
				reduced_grad(tiled_to_single_(i_full) * dim + d) += grad(i_full * dim + d);

			for (int p = 0; p < dim; p++)
				for (int q = 0; q < dim; q++)
					reduced_grad(n_single_dof_ * dim + p * dim + q) += boundary_vertices(i, q) * grad(i_full * dim + p);
		}

		return reduced_grad;
	}

	void PeriodicContactForm::init(const Eigen::VectorXd &x)
	{
		BarrierContactForm::init(single_to_tiled(x));
	}

	double PeriodicContactForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		return BarrierContactForm::value_unweighted(single_to_tiled(x));
	}

	void PeriodicContactForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		BarrierContactForm::first_derivative_unweighted(single_to_tiled(x), gradv);
		gradv = tiled_to_single_grad(gradv);
	}

	void PeriodicContactForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		StiffnessMatrix hessian_full;
		BarrierContactForm::second_derivative_unweighted(single_to_tiled(x), hessian_full);

		update_projection();
		hessian = proj * hessian_full * proj.transpose();

		// const Eigen::MatrixXd displaced = collision_mesh_.displace_vertices(utils::unflatten(single_to_tiled(x), collision_mesh_.dim()));

		// io::OBJWriter::write(
		//     "tiled.obj", displaced,
		//     collision_mesh_.edges(), collision_mesh_.faces());
	}

	void PeriodicContactForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		BarrierContactForm::update_quantities(t, single_to_tiled(x));
	}

	double PeriodicContactForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return BarrierContactForm::max_step_size(single_to_tiled(x0), single_to_tiled(x1));
	}

	void PeriodicContactForm::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		BarrierContactForm::line_search_begin(single_to_tiled(x0), single_to_tiled(x1));
	}

	void PeriodicContactForm::solution_changed(const Eigen::VectorXd &new_x)
	{
		BarrierContactForm::solution_changed(single_to_tiled(new_x));
	}

	void PeriodicContactForm::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		BarrierContactForm::post_step(
			polysolve::nonlinear::PostStepData(
				data.iter_num, data.solver_info, single_to_tiled(data.x), single_to_tiled(data.grad)));
	}

	bool PeriodicContactForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		return BarrierContactForm::is_step_collision_free(single_to_tiled(x0), single_to_tiled(x1));
	}

	void PeriodicContactForm::update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy)
	{
		BarrierContactForm::update_barrier_stiffness(single_to_tiled(x), single_to_tiled(grad_energy));
	}
} // namespace polyfem::solver

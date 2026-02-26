#include "PeriodicContactForceDerivative.hpp"

#include <Eigen/Core>
#include <polyfem/solver/forms/PeriodicContactForm.hpp>
#include <polyfem/State.hpp>
#include <polyfem/optimization/parametrization/PeriodicMeshToMesh.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/Types.hpp>
#include <ipc/collisions/normal/normal_collisions.hpp>

namespace polyfem::solver
{
	void PeriodicContactForceDerivative::force_shape_derivative(
		const PeriodicContactForm &form,
		const State &state,
		const PeriodicMeshToMesh &periodic_mesh_map,
		const Eigen::VectorXd &periodic_mesh_representation,
		const ipc::NormalCollisions &contact_set,
		const Eigen::VectorXd &solution,
		const Eigen::VectorXd &adjoint_sol,
		Eigen::VectorXd &term)
	{
		const int dim = form.collision_mesh_.dim();
		const Eigen::MatrixXd displaced_surface = form.compute_displaced_surface(form.single_to_tiled(solution));

		Eigen::VectorXd tiled_term;

		{
			StiffnessMatrix dq_h = form.collision_mesh_.to_full_dof(form.barrier_potential().shape_derivative(contact_set, form.collision_mesh_, displaced_surface));
			tiled_term = dq_h.transpose() * form.single_to_tiled(adjoint_sol);
		}

		{
			Eigen::VectorXd force;
			force = form.barrier_potential().gradient(contact_set, form.collision_mesh_, displaced_surface);
			force = form.collision_mesh_.to_full_dof(force);
			Eigen::MatrixXd adjoint_affine = utils::unflatten(adjoint_sol.tail(dim * dim), dim);
			for (int k = 0; k < form.collision_mesh_.num_vertices(); k++)
			{
				const int k_full = form.collision_mesh_.to_full_vertex_id(k);
				tiled_term.segment(k_full * dim, dim) += adjoint_affine.transpose() * force.segment(k_full * dim, dim);
			}
		}

		{
			StiffnessMatrix hessian_full;
			form.BarrierContactForm::second_derivative_unweighted(form.single_to_tiled(solution), hessian_full);
			Eigen::VectorXd tmp = form.single_to_tiled(adjoint_sol).transpose() * hessian_full;
			Eigen::MatrixXd sol_affine = utils::unflatten(solution.tail(dim * dim), dim);
			for (int k = 0; k < form.collision_mesh_.num_vertices(); k++)
			{
				const int k_full = form.collision_mesh_.to_full_vertex_id(k);
				tiled_term.segment(k_full * dim, dim) += sol_affine.transpose() * tmp.segment(k_full * dim, dim);
			}
		}

		// chain rule from tiled to periodic
		Eigen::VectorXd unit_term;
		Eigen::MatrixXd affine_term;
		affine_term.setZero(dim, dim);
		unit_term.setZero(tiled_term.size());
		Eigen::MatrixXd affine = utils::unflatten(periodic_mesh_representation.tail(dim * dim), dim).transpose();
		for (int k = 0; k < form.collision_mesh_.num_vertices(); k++)
		{
			const int k_full = form.collision_mesh_.to_full_vertex_id(k);
			affine_term += tiled_term.segment(k_full * dim, dim) * form.collision_mesh_.rest_positions().row(k);
			unit_term.segment(k_full * dim, dim) += affine.transpose() * tiled_term.segment(k_full * dim, dim);
		}

		Eigen::VectorXd single_term = state.basis_nodes_to_gbasis_nodes * form.proj.topRows(dim * form.n_single_dof_) * unit_term;
		single_term = utils::flatten(utils::unflatten(single_term, dim)(state.primitive_to_node(), Eigen::all));

		term.setZero(periodic_mesh_representation.size());
		for (int i = 0; i < state.n_geom_bases; i++)
			term.segment(periodic_mesh_map.full_to_periodic(i) * dim, dim).array() += single_term.segment(i * dim, dim).array();
		term.tail(dim * dim) = Eigen::Map<Eigen::VectorXd>(affine_term.data(), dim * dim, 1);

		term *= form.weight();
	}
} // namespace polyfem::solver

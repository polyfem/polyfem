#include <polyfem/legacy/State.hpp>

#include <polyfem/assembler/Laplacian.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/Stokes.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>
#include <polyfem/solver/OperatorSplittingSolver.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>

#include <cassert>
#include <vector>

namespace polyfem::legacy
{
	void State::solve_transient_navier_stokes_split(
		const int time_steps,
		const double dt,
		Eigen::MatrixXd &sol,
		Eigen::MatrixXd &pressure,
		UserPostStepCallback user_post_step)
	{
		assert(assembler->name() == "OperatorSplitting" && problem->is_time_dependent());

		Eigen::MatrixXd local_pts;
		auto &gbases = geom_bases();
		if (mesh->dimension() == 2)
		{
			if (gbases[0].bases.size() == 3)
				autogen::p_nodes_2d(args["space"]["discr_order"], local_pts);
			else
				autogen::q_nodes_2d(args["space"]["discr_order"], local_pts);
		}
		else
		{
			if (gbases[0].bases.size() == 4)
				autogen::p_nodes_3d(args["space"]["discr_order"], local_pts);
			else
				autogen::q_nodes_3d(args["space"]["discr_order"], local_pts);
		}

		std::vector<int> bnd_nodes;
		bnd_nodes.reserve(boundary_nodes.size() / mesh->dimension());
		for (const int node : boundary_nodes)
		{
			if (node % mesh->dimension() == 0)
				continue;
			bnd_nodes.push_back(node / mesh->dimension());
		}

		const int n_el = int(bases.size());
		const int shape = gbases[0].bases.size();
		auto fluid_assembler = std::dynamic_pointer_cast<assembler::OperatorSplitting>(assembler);
		if (!fluid_assembler)
			log_and_throw_error("Invalid assembler {}!", assembler->name());
		const double viscosity = fluid_assembler->viscosity()(0, 0, 0, 0, 0);
		assert(viscosity >= 0);

		logger().info("Matrices assembly...");
		StiffnessMatrix stiffness_viscosity, mixed_stiffness, velocity_mass, stiffness;
		build_stiffness_mat(stiffness);

		assembler::Laplacian laplacian;
		laplacian.set_size(1);
		laplacian.assemble(mesh->is_volume(), n_bases, bases, gbases, ass_vals_cache, 0, stiffness_viscosity);
		mass_matrix_assembler->set_size(1);
		mass_matrix_assembler->assemble(mesh->is_volume(), n_bases, bases, gbases, mass_ass_vals_cache, 0, mass, true);
		laplacian.assemble(mesh->is_volume(), n_pressure_bases, pressure_bases, gbases, pressure_ass_vals_cache, 0, stiffness);

		mixed_assembler->assemble(
			mesh->is_volume(), n_pressure_bases, n_bases,
			pressure_bases, bases, gbases,
			pressure_ass_vals_cache, ass_vals_cache, 0, mixed_stiffness);
		mass_matrix_assembler->set_size(mesh->dimension());
		mass_matrix_assembler->assemble(mesh->is_volume(), n_bases, bases, gbases, mass_ass_vals_cache, 0, velocity_mass, true);
		mixed_stiffness = mixed_stiffness.transpose();
		logger().info("Matrices assembly ends!");

		solver::OperatorSplittingSolver splitting_solver(
			*mesh, shape, n_el, local_boundary, boundary_nodes,
			pressure_boundary_nodes, bnd_nodes, mass,
			stiffness_viscosity, stiffness, velocity_mass,
			dt, viscosity, args["solver"]["linear"]);

		pressure = Eigen::MatrixXd::Zero(n_pressure_bases, 1);
		if (user_post_step)
			user_post_step(0, *this, sol, nullptr, &pressure);

		const QuadratureOrders &boundary_samples = n_boundary_samples();
		for (int step = 1; step <= time_steps; ++step)
		{
			const double time = step * dt;
			logger().info("{}/{} steps, t={}s", step, time_steps, time);

			if (args["space"]["advanced"]["use_particle_advection"])
				splitting_solver.advection_FLIP(*mesh, gbases, bases, sol, dt, local_pts);
			else
				splitting_solver.advection(*mesh, gbases, bases, sol, dt, local_pts);

			solve_data.rhs_assembler->set_bc(
				local_boundary, boundary_nodes, boundary_samples,
				local_neumann_boundary, sol, Eigen::MatrixXd(), time);

			if (viscosity > 0)
				splitting_solver.solve_diffusion_1st(mass, bnd_nodes, sol);
			splitting_solver.external_force(
				*mesh, *assembler, gbases, bases, dt, sol, local_pts, problem, time);
			splitting_solver.solve_pressure(mixed_stiffness, pressure_boundary_nodes, sol, pressure);
			splitting_solver.projection(
				n_bases, gbases, bases, pressure_bases, local_pts, pressure, sol);
			pressure /= dt;

			solve_data.rhs_assembler->set_bc(
				local_boundary, boundary_nodes, boundary_samples,
				local_neumann_boundary, sol, Eigen::MatrixXd(), time);

			save_timestep(time, step, 0, dt, sol, pressure);
			if (user_post_step)
				user_post_step(step, *this, sol, nullptr, &pressure);
		}
	}
} // namespace polyfem::legacy

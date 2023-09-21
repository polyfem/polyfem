#include <polyfem/State.hpp>

#include <polyfem/assembler/Laplacian.hpp>
#include <polyfem/assembler/Mass.hpp>
#include <polyfem/assembler/Stokes.hpp>
#include <polyfem/assembler/NavierStokes.hpp>

#include <polyfem/solver/NavierStokesSolver.hpp>
#include <polyfem/solver/OperatorSplittingSolver.hpp>
#include <polyfem/solver/TransientNavierStokesSolver.hpp>
#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

namespace polyfem
{
	using namespace solver;
	using namespace time_integrator;

	void State::solve_navier_stokes(Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
	{
		assert(!problem->is_time_dependent());
		assert(assembler->name() == "NavierStokes");

		assert(solve_data.rhs_assembler != nullptr);
		solve_data.rhs_assembler->set_bc(
			local_boundary, boundary_nodes, n_boundary_samples(), local_neumann_boundary, rhs);

		std::shared_ptr<assembler::Assembler> velocity_stokes_assembler = std::make_shared<assembler::StokesVelocity>();
		set_materials(*velocity_stokes_assembler);

		Eigen::VectorXd x;
		solver::NavierStokesSolver ns_solver(args["solver"]);
		ns_solver.minimize(n_bases, n_pressure_bases,
						   bases, pressure_bases,
						   geom_bases(),
						   *velocity_stokes_assembler,
						   *dynamic_cast<assembler::NavierStokesVelocity *>(assembler.get()),
						   *mixed_assembler,
						   *pressure_assembler,
						   ass_vals_cache,
						   pressure_ass_vals_cache,
						   boundary_nodes,
						   use_avg_pressure,
						   mesh->dimension(),
						   mesh->is_volume(),
						   rhs, x);

		sol = x;
		sol_to_pressure(sol, pressure);
	}

	void State::solve_transient_navier_stokes_split(const int time_steps, const double dt, Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
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
		for (auto it = boundary_nodes.begin(); it != boundary_nodes.end(); it++)
		{
			if (!(*it % mesh->dimension()))
				continue;
			bnd_nodes.push_back(*it / mesh->dimension());
		}

		const int dim = mesh->dimension();
		const int n_el = int(bases.size());       // number of elements
		const int shape = gbases[0].bases.size(); // number of geometry vertices in an element
		
		double viscosity_ = std::dynamic_pointer_cast<assembler::OperatorSplitting>(assembler)->viscosity();
		assert(viscosity_ >= 0);

		logger().info("Matrices assembly...");
		StiffnessMatrix stiffness_viscosity, mixed_stiffness, velocity_mass, stiffness;

		build_stiffness_mat(stiffness);
		// coefficient matrix of viscosity
		assembler::Laplacian lapl_assembler;
		lapl_assembler.set_size(1);
		lapl_assembler.assemble(mesh->is_volume(), n_bases, bases, gbases, ass_vals_cache, stiffness_viscosity);
		mass_matrix_assembler->set_size(1);
		mass_matrix_assembler->assemble(mesh->is_volume(), n_bases, bases, gbases, mass_ass_vals_cache, mass, true);

		// coefficient matrix of pressure projection
		lapl_assembler.assemble(mesh->is_volume(), n_pressure_bases, pressure_bases, gbases, pressure_ass_vals_cache, stiffness);

		// matrix used to calculate divergence of velocity
		mixed_assembler->assemble(mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, gbases,
								  pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
		mass_matrix_assembler->set_size(mesh->dimension());
		mass_matrix_assembler->assemble(mesh->is_volume(), n_bases, bases, gbases, mass_ass_vals_cache, velocity_mass, true);
		mixed_stiffness = mixed_stiffness.transpose();
		logger().info("Matrices assembly ends!");

		solver::OperatorSplittingSolver ss(
			*mesh, shape, n_el, local_boundary, boundary_nodes, pressure_boundary_nodes, bnd_nodes, mass,
			stiffness_viscosity, stiffness, velocity_mass, dt, viscosity_, args["solver"]["linear"]["solver"],
			args["solver"]["linear"]["precond"], args["solver"]["linear"]);

		/* initialize solution */
		pressure = Eigen::MatrixXd::Zero(n_pressure_bases, 1);

		const int n_b_samples = n_boundary_samples();
		for (int t = 1; t <= time_steps; t++)
		{
			double time = t * dt;
			logger().info("{}/{} steps, t={}s", t, time_steps, time);

			/* advection */
			logger().info("Advection...");
			if (args["space"]["advanced"]["use_particle_advection"])
				ss.advection_FLIP(*mesh, gbases, bases, sol, dt, local_pts);
			else
				ss.advection(*mesh, gbases, bases, sol, dt, local_pts);
			logger().info("Advection finished!");

			/* apply boundary condition */
			solve_data.rhs_assembler->set_bc(
				local_boundary, boundary_nodes, n_b_samples, local_neumann_boundary, sol, Eigen::MatrixXd(), time);

			/* viscosity */
			logger().info("Solving diffusion...");
			if (viscosity_ > 0)
				ss.solve_diffusion_1st(mass, bnd_nodes, sol);
			logger().info("Diffusion solved!");

			/* external force */
			ss.external_force(*mesh, *assembler, gbases, bases, dt, sol, local_pts, problem, time);

			/* incompressibility */
			logger().info("Pressure projection...");
			ss.solve_pressure(mixed_stiffness, pressure_boundary_nodes, sol, pressure);

			ss.projection(n_bases, gbases, bases, pressure_bases, local_pts, pressure, sol);
			// ss.projection(velocity_mass, mixed_stiffness, boundary_nodes, sol, pressure);
			logger().info("Pressure projection finished!");

			pressure = pressure / dt;

			/* apply boundary condition */
			solve_data.rhs_assembler->set_bc(
				local_boundary, boundary_nodes, n_b_samples, local_neumann_boundary, sol, Eigen::MatrixXd(), time);

			/* export to vtu */
			save_timestep(time, t, 0, dt, sol, pressure);
		}
	}

	void State::solve_transient_navier_stokes(const int time_steps, const double t0, const double dt, Eigen::MatrixXd &sol, Eigen::MatrixXd &pressure)
	{
		assert(assembler->name() == "NavierStokes" && problem->is_time_dependent());

		const auto &gbases = geom_bases();
		Eigen::MatrixXd current_rhs = rhs;

		StiffnessMatrix velocity_mass;
		mass_matrix_assembler->assemble(mesh->is_volume(), n_bases, bases, gbases, mass_ass_vals_cache, velocity_mass, true);

		StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;

		Eigen::VectorXd prev_sol;

		BDF time_integrator;
		if (args["time"]["integrator"].is_object() && args["time"]["integrator"]["type"] == "BDF")
			time_integrator.set_parameters(args["time"]["integrator"]);
		time_integrator.init(sol, Eigen::VectorXd::Zero(sol.size()), Eigen::VectorXd::Zero(sol.size()), dt);

		std::shared_ptr<assembler::Assembler> velocity_stokes_assembler = std::make_shared<assembler::StokesVelocity>();
		set_materials(*velocity_stokes_assembler);

		velocity_stokes_assembler->assemble(mesh->is_volume(), n_bases, bases, gbases, ass_vals_cache, velocity_stiffness);
		mixed_assembler->assemble(mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, gbases,
								  pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
		pressure_assembler->assemble(mesh->is_volume(), n_pressure_bases, pressure_bases, gbases, pressure_ass_vals_cache,
									 pressure_stiffness);

		solver::TransientNavierStokesSolver ns_solver(args["solver"]);
		const int n_larger = n_pressure_bases + (use_avg_pressure ? 1 : 0);

		const int n_b_samples = n_boundary_samples();

		for (int t = 1; t <= time_steps; ++t)
		{
			double time = t0 + t * dt;
			double current_dt = dt;

			logger().info("{}/{} steps, dt={}s t={}s", t, time_steps, current_dt, time);

			prev_sol = time_integrator.weighted_sum_x_prevs();
			solve_data.rhs_assembler->compute_energy_grad(
				local_boundary, boundary_nodes, mass_matrix_assembler->density(), n_b_samples, local_neumann_boundary, rhs, time, current_rhs);
			solve_data.rhs_assembler->set_bc(
				local_boundary, boundary_nodes, n_b_samples, local_neumann_boundary, current_rhs, Eigen::MatrixXd(), time);

			const int prev_size = current_rhs.size();
			if (prev_size != rhs.size())
			{
				current_rhs.conservativeResize(prev_size + n_larger, current_rhs.cols());
				current_rhs.block(prev_size, 0, n_larger, current_rhs.cols()).setZero();
			}

			Eigen::VectorXd tmp_sol;
			ns_solver.minimize(
				n_bases, n_pressure_bases,
				bases, geom_bases(),
				*dynamic_cast<assembler::NavierStokesVelocity *>(assembler.get()),
				ass_vals_cache,
				boundary_nodes,
				use_avg_pressure,
				mesh->dimension(),
				mesh->is_volume(),
				sqrt(time_integrator.acceleration_scaling()), prev_sol, velocity_stiffness, mixed_stiffness,
				pressure_stiffness, velocity_mass, current_rhs, tmp_sol);
			sol = tmp_sol;
			time_integrator.update_quantities(sol.topRows(n_bases * mesh->dimension()));
			sol_to_pressure(sol, pressure);

			save_timestep(time, t, t0, dt, sol, pressure);
		}
	}
} // namespace polyfem

#pragma once

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

namespace polyfem::mesh
{
	struct RemesherTimings
	{
		double total = 0;
		double create_local_mesh = 0;
		double build_bases = 0;
		double create_boundary_nodes = 0;
		double create_assembler = 0;
		double assemble_mass_matrix = 0;
		double create_collision_mesh = 0;
		double init_forms = 0;
		double local_relaxation_solve = 0;
		double acceptance_check = 0;
		int total_ndofs = 0;
		int n_solves = 0;

		void reset()
		{
			total = 0;
			create_local_mesh = 0;
			build_bases = 0;
			create_boundary_nodes = 0;
			create_assembler = 0;
			assemble_mass_matrix = 0;
			create_collision_mesh = 0;
			init_forms = 0;
			local_relaxation_solve = 0;
			acceptance_check = 0;
		}

		void log()
		{
			logger().critical("Total time: {:.3g}s", total);
			logger().critical("Create local mesh: {:.3g}s {:.1f}%", create_local_mesh, (create_local_mesh) / total * 100);
			logger().critical("Build bases: {:.3g}s {:.1f}%", build_bases, (build_bases) / total * 100);
			logger().critical("Create boundary nodes: {:.3g}s {:.1f}%", create_boundary_nodes, (create_boundary_nodes) / total * 100);
			logger().critical("Create assembler: {:.3g}s {:.1f}%", create_assembler, (create_assembler) / total * 100);
			logger().critical("Assemble mass matrix: {:.3g}s {:.1f}%", assemble_mass_matrix, (assemble_mass_matrix) / total * 100);
			logger().critical("Create collision mesh: {:.3g}s {:.1f}%", create_collision_mesh, (create_collision_mesh) / total * 100);
			logger().critical("Init forms: {:.3g}s {:.1f}%", init_forms, (init_forms) / total * 100);
			logger().critical("Local relaxation solve: {:.3g}s {:.1f}%", local_relaxation_solve, (local_relaxation_solve) / total * 100);
			logger().critical("Acceptance check: {:.3g}s {:.1f}%", acceptance_check, (acceptance_check) / total * 100);
			logger().critical("Miscellaneous: {:.3g}s {:.1f}%", total - sum(), (total - sum()) / total * 100);
			if (n_solves > 0)
				logger().critical("Avg. # DOF per solve: {}", total_ndofs / double(n_solves));
		}

		double sum()
		{
			return create_local_mesh + build_bases + create_boundary_nodes
				   + create_assembler + assemble_mass_matrix + create_collision_mesh
				   + init_forms + local_relaxation_solve + acceptance_check;
		}
	};

} // namespace polyfem::mesh

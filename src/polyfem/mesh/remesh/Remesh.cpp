#include "Remesh.hpp"

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALNLProblem.hpp>
#include <polyfem/mesh/remesh/L2Projection.hpp>
#include <polyfem/mesh/remesh/MMGRemesh.hpp>
#include <polyfem/utils/OBJ_IO.hpp>

#include <igl/PI.h>

namespace polyfem::mesh
{
	using namespace utils;

	void remesh(State &state, const double t0, const double dt, const int t)
	{
		Eigen::MatrixXd V(state.mesh->n_vertices(), state.mesh->dimension());
		for (int i = 0; i < state.mesh->n_vertices(); ++i)
			V.row(i) = state.mesh->point(i);
		Eigen::MatrixXi F(state.mesh->n_faces(), state.mesh->dimension() + 1);
		for (int i = 0; i < F.rows(); ++i)
			for (int j = 0; j < F.cols(); ++j)
				F(i, j) = state.mesh->face_vertex(i, j);
		OBJWriter::save(state.resolve_output_path("rest.obj"), V, F);

		// TODO: compute stress at the nodes
		// Eigen::MatrixXd SF;
		// compute_scalar_value(state.mesh->n_vertices(), sol, SF, false, false);
		Eigen::MatrixXd SV, TV;
		// average_grad_based_function(state.mesh->n_vertices(), sol, SV, TV, false, false);

		// TODO: What measure to use for remeshing?
		// SV.normalize();
		SV.setOnes(state.mesh->n_vertices(), 1);
		SV *= 0.1 / t;

		MmgOptions mmg_options;
		mmg_options.hmin = 1e-4;

		Eigen::MatrixXd V_new;
		Eigen::MatrixXi F_new;
		if (!state.mesh->is_volume())
		{
			remesh_adaptive_2d(V, F, SV, V_new, F_new, mmg_options);

			// Rotate 90 degrees each step
			// Matrix2d R;
			// const double theta = 90 * (igl::PI / 180);
			// R << cos(theta), sin(theta),
			// 	-sin(theta), cos(theta);
			// V_new = V * R.transpose();

			// V_new = V;
			// F_new = F;

			OBJWriter::save(state.resolve_output_path("remeshed.obj"), V_new, F_new);
		}
		else
		{
			Eigen::MatrixXi _;
			remesh_adaptive_3d(V, F, SV, V_new, _, F_new);
		}

		// --------------------------------------------------------------------

		// Save old values
		const int old_n_bases = state.n_bases;
		const std::vector<ElementBases> old_bases = state.bases;
		// TODO: replace with state.geom_bases()
		const std::vector<ElementBases> old_geom_bases = state.iso_parametric() ? state.bases : state.geom_bases;
		const StiffnessMatrix old_mass = state.mass;
		Eigen::MatrixXd y(state.sol.size(), 3); // Old values of independent variables
		y.col(0) = state.sol;
		y.col(1) = state.solve_data.nl_problem->time_integrator()->v_prev();
		y.col(2) = state.solve_data.nl_problem->time_integrator()->a_prev();

		// --------------------------------------------------------------------

		state.load_mesh(V_new, F_new);
		// FIXME:
		state.mesh->compute_boundary_ids(1e-6);
		state.mesh->set_body_ids(std::vector<int>(state.mesh->n_elements(), 1));
		state.set_materials(); // TODO: Explain why I need this?
		state.build_basis();
		state.assemble_rhs();
		state.assemble_stiffness_mat();

		// --------------------------------------------------------------------

		// L2 Projection
		state.ass_vals_cache.clear(); // Clear this because the mass matrix needs to be recomputed
		Eigen::MatrixXd x;
		L2_projection(
			state, *state.solve_data.rhs_assembler,
			state.mesh->is_volume(), state.mesh->is_volume() ? 3 : 2,
			old_n_bases, old_bases, old_geom_bases,                                              // from
			state.n_bases, state.bases, state.iso_parametric() ? state.bases : state.geom_bases, // to // TODO: replace with state.geom_bases()
			state.ass_vals_cache, y, x, /*lump_mass_matrix=*/false);

		state.sol = x.col(0);
		Eigen::VectorXd vel = x.col(1);
		Eigen::VectorXd acc = x.col(2);

		if (x.rows() < 30)
		{
			logger().critical("yᵀ:\n{}", y.transpose());
			logger().critical("xᵀ:\n{}", x.transpose());
		}

		// Compute Projection error
		if (false)
		{
			polyfem::assembler::AssemblyValsCache ass_vals_cache; // TODO: Init this?
			Eigen::MatrixXd y2;
			L2_projection(
				state, *state.solve_data.rhs_assembler,
				state.mesh->is_volume(), state.mesh->is_volume() ? 3 : 2,
				state.n_bases, state.bases, state.iso_parametric() ? state.bases : state.geom_bases, // from // TODO: replace with state.geom_bases()
				old_n_bases, old_bases, old_geom_bases,                                              // to
				ass_vals_cache, x, y2);

			auto error = [&old_mass](const Eigen::VectorXd &old_y, const Eigen::VectorXd &new_y) -> double {
				const auto diff = new_y - old_y;
				return diff.transpose() * old_mass * diff;
				// return diff.norm() / diff.rows();
			};

			std::cout << fmt::format(
				"L2_Projection_Error, {}, {}, {}, {}",
				error(y.col(0), y2.col(0)),
				error(y.col(1), y2.col(1)),
				error(y.col(2), y2.col(2)),
				V_new.rows())
					  << std::endl;
		}

		// --------------------------------------------------------------------

		// TODO: Replace with state.build_rhs_assembler()
		json rhs_solver_params = state.args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)
		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		state.solve_data.rhs_assembler = std::make_shared<RhsAssembler>(
			state.assembler, *state.mesh, state.obstacle, state.input_dirichlet,
			state.n_bases, state.problem->is_scalar() ? 1 : state.mesh->dimension(),
			state.bases, gbases, state.ass_vals_cache,
			state.formulation(), *state.problem,
			state.args["space"]["advanced"]["bc_method"],
			state.args["solver"]["linear"]["solver"],
			state.args["solver"]["linear"]["precond"],
			rhs_solver_params);

		const int full_size = state.n_bases * state.mesh->dimension();
		const int reduced_size = state.n_bases * state.mesh->dimension() - state.boundary_nodes.size();

		state.solve_data.nl_problem = std::make_shared<NLProblem>(
			state, *state.solve_data.rhs_assembler, t0 + t * dt, state.args["contact"]["dhat"]);
		state.solve_data.nl_problem->init_time_integrator(state.sol, vel, acc, dt);

		double al_weight = state.args["solver"]["augmented_lagrangian"]["initial_weight"];
		state.solve_data.alnl_problem = std::make_shared<ALNLProblem>(
			state, *state.solve_data.rhs_assembler, t0 + t * dt, state.args["contact"]["dhat"], al_weight);
		state.solve_data.alnl_problem->init_time_integrator(state.sol, vel, acc, dt);

		// TODO: Check for inversions and intersections due to remeshing
	}
} // namespace polyfem::mesh

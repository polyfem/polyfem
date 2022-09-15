#include "Remesh.hpp"

#include <polyfem/mesh/remesh/L2Projection.hpp>
#include <polyfem/mesh/remesh/MMGRemesh.hpp>
#include <polyfem/mesh/remesh/WildRemesh2D.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/io/OBJWriter.hpp>

#include <igl/PI.h>
#include <igl/boundary_facets.h>

namespace polyfem::mesh
{
	using namespace time_integrator;
	using namespace io;
	using namespace utils;

	void remesh(State &state, const double t0, const double dt, const int t)
	{
		const ImplicitTimeIntegrator &time_integrator = *state.solve_data.time_integrator;

		Eigen::MatrixXd V(state.mesh->n_vertices(), state.mesh->dimension());
		for (int i = 0; i < state.mesh->n_vertices(); ++i)
			V.row(i) = state.mesh->point(i);

		Eigen::MatrixXi F(state.mesh->n_elements(), state.mesh->dimension() + 1);
		for (int i = 0; i < F.rows(); ++i)
			for (int j = 0; j < F.cols(); ++j)
				F(i, j) = state.mesh->element_vertex(i, j);

		if (!state.mesh->is_volume())
			OBJWriter::write(state.resolve_output_path("rest.obj"), V, F);
		else
		{
			Eigen::MatrixXi BF;
			igl::boundary_facets(F, BF);
			OBJWriter::write(state.resolve_output_path("rest.obj"), V, BF);
		}

		Eigen::MatrixXd V_new;
		Eigen::MatrixXi F_new;

#ifdef USE_MMG_REMESHING
		MmgOptions mmg_options;
		mmg_options.hmin = 1e-4;
		if (!state.mesh->is_volume())
		{
			// TODO: What measure to use for remeshing?
			Eigen::MatrixXd SV;
			SV.setOnes(V.rows(), 1);
			SV *= 0.1 / t;

			remesh_adaptive_2d(V, F, SV, V_new, F_new, mmg_options);

			// Rotate 90 degrees each step
			// Matrix2d R;
			// const double theta = 90 * (igl::PI / 180);
			// R << cos(theta), sin(theta),
			// 	-sin(theta), cos(theta);
			// V_new = V * R.transpose();

			// V_new = V;
			// F_new = F;

			OBJWriter::write(state.resolve_output_path("remeshed.obj"), V_new, F_new);
		}
		else
		{
			// TODO: What measure to use for remeshing?
			Eigen::MatrixXd SV;
			SV.setOnes(V.rows(), 1);
			SV *= 0.1 / t;

			Eigen::MatrixXi BF_new;
			remesh_adaptive_3d(V, F, SV, V_new, BF_new, F_new);
			OBJWriter::write(state.resolve_output_path("remeshed.obj"), V_new, BF_new);
		}
#else
		const int n_vertices = state.mesh->n_vertices();
		const int dim = state.mesh->dimension();
		Eigen::MatrixXd U = unflatten(state.sol, dim);
		Eigen::MatrixXd Vel = unflatten(time_integrator.v_prev(), dim);
		Eigen::MatrixXd Acc = unflatten(time_integrator.a_prev(), dim);
		assert(!state.mesh->is_volume());
		WildRemeshing2D remeshing;
		remeshing.create_mesh(V, F, U, Vel, Acc);
		// for(int i = 0; i < 10; ++i)
		{
			remeshing.smooth_all_vertices();
			// remeshing.split_all_edges();
		}
		remeshing.export_mesh(V_new, F_new, U, Vel, Acc);
		// TODO: use U, Vel, Acc
		// state.sol = flatten(U);
		// return;
#endif

		// --------------------------------------------------------------------

		// Save old values
		const int old_n_bases = state.n_bases;
		const std::vector<ElementBases> old_bases = state.bases;
		const std::vector<ElementBases> old_geom_bases = state.geom_bases();
		const StiffnessMatrix old_mass = state.mass;
		Eigen::MatrixXd y(state.sol.size(), 3); // Old values of independent variables
		y.col(0) = state.sol;
		y.col(1) = time_integrator.v_prev();
		y.col(2) = time_integrator.a_prev();

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
			old_n_bases, old_bases, old_geom_bases,         // from
			state.n_bases, state.bases, state.geom_bases(), // to
			state.ass_vals_cache, y, x, t0, dt, t, /*lump_mass_matrix=*/false);

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
				state.n_bases, state.bases, state.geom_bases(), // from
				old_n_bases, old_bases, old_geom_bases,         // to
				ass_vals_cache, x, y2, t0, dt, t);

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

		state.solve_data.rhs_assembler = state.build_rhs_assembler();
		state.init_nonlinear_tensor_solve(t0 + t * dt);
		if (state.problem->is_time_dependent())
		{
			state.solve_data.time_integrator->init(state.sol, vel, acc, dt);
		}

		// TODO: Check for inversions and intersections due to remeshing
	}
} // namespace polyfem::mesh

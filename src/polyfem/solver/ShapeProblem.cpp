#include <polyfem/ShapeProblem.hpp>

#include <polyfem/Types.hpp>
#include <polyfem/Timer.hpp>
#include <polyfem/MatrixUtils.hpp>

#include <igl/writeOBJ.h>
#include <igl/writeMESH.h>
#include <igl/write_triangle_mesh.h>
#include <igl/avg_edge_length.h>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>

#include <filesystem>

namespace polyfem
{
	namespace
	{
		double triangle_jacobian(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3)
		{
			Eigen::VectorXd a = v2 - v1, b = v3 - v1;
			return a(0) * b(1) - b(0) * a(1);
		}

		double tet_determinant(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4)
		{
			Eigen::Matrix3d mat;
			mat.col(0) << v2 - v1;
			mat.col(1) << v3 - v1;
			mat.col(2) << v4 - v1;
			return mat.determinant();
		}

		bool is_flipped(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
		{
			if (F.cols() == 3)
			{
				for (int i = 0; i < F.rows(); i++)
					if (triangle_jacobian(V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2))) <= 0)
						return true;
			}
			else if (F.cols() == 4)
			{
				for (int i = 0; i < F.rows(); i++)
					if (tet_determinant(V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2)), V.row(F(i, 3))) <= 0)
						return true;
			}
			else
			{
				return true;
			}

			return false;
		}

		void scaled_jacobian(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Eigen::VectorXd &quality)
		{
			const int dim = F.cols() - 1;
			assert(V.cols() == 3);

			quality.setZero(F.rows());
			if (dim == 2)
			{
				for (int i = 0; i < F.rows(); i++)
				{
					Eigen::RowVector3d e0 = V.row(F(i, 2)) - V.row(F(i, 1));
					Eigen::RowVector3d e1 = V.row(F(i, 0)) - V.row(F(i, 2));
					Eigen::RowVector3d e2 = V.row(F(i, 1)) - V.row(F(i, 0));

					double l0 = e0.norm();
					double l1 = e1.norm();
					double l2 = e2.norm();

					double A = 0.5 * (e0.cross(e1)).norm();
					double Lmax = std::max(l0 * l1, std::max(l1 * l2, l0 * l2));

					quality(i) = 2 * A * (2 / sqrt(3)) / Lmax;
				}
			}
			else
			{
				for (int i = 0; i < F.rows(); i++)
				{
					Eigen::RowVector3d e0 = V.row(F(i, 1)) - V.row(F(i, 0));
					Eigen::RowVector3d e1 = V.row(F(i, 2)) - V.row(F(i, 1));
					Eigen::RowVector3d e2 = V.row(F(i, 0)) - V.row(F(i, 2));
					Eigen::RowVector3d e3 = V.row(F(i, 3)) - V.row(F(i, 0));
					Eigen::RowVector3d e4 = V.row(F(i, 3)) - V.row(F(i, 1));
					Eigen::RowVector3d e5 = V.row(F(i, 3)) - V.row(F(i, 2));

					double l0 = e0.norm();
					double l1 = e1.norm();
					double l2 = e2.norm();
					double l3 = e3.norm();
					double l4 = e4.norm();
					double l5 = e5.norm();

					double J = std::abs((e0.cross(e3)).dot(e2));

					double a1 = l0 * l2 * l3;
					double a2 = l0 * l1 * l4;
					double a3 = l1 * l2 * l5;
					double a4 = l3 * l4 * l5;

					double a = std::max({a1, a2, a3, a4, J});
					quality(i) = J * sqrt(2) / a;
				}
			}
		}

		bool internal_smoothing(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const std::vector<int> &boundary_indices, const Eigen::MatrixXd &boundary_constraints, const json &slim_params, Eigen::MatrixXd &smooth_field)
		{
			const int dim = F.cols() - 1;
			igl::SLIMData slim_data;
			double soft_const_p = slim_params["soft_p"];
			slim_data.exp_factor = slim_params["exp_factor"];
			Eigen::MatrixXd V_extended;
			V_extended.setZero(V.rows(), 3);
			V_extended.block(0, 0, V.rows(), dim) = V;
			Eigen::VectorXi boundary_indices_ = Eigen::VectorXi::Map(boundary_indices.data(), boundary_indices.size());
			igl::slim_precompute(
				V_extended,
				F,
				V,
				slim_data,
				igl::SYMMETRIC_DIRICHLET,
				boundary_indices_,
				boundary_constraints,
				soft_const_p);

			smooth_field.setZero(V.rows(), V.cols());

			auto is_good_enough = [](const Eigen::MatrixXd &V, const Eigen::VectorXi &b, const Eigen::MatrixXd &C, double &error, double tol = 1e-5) {
				error = 0.0;

				for (unsigned i = 0; i < b.rows(); i++)
					error += (C.row(i) - V.row(b(i))).squaredNorm();

				return error < tol;
			};

			double error = 0;
			int max_it = dim == 2 ? 20 : 50;
			int it = 0;
			bool good_enough = false;

			do
			{
				igl::slim_solve(slim_data, slim_params["min_iter"]);
				good_enough = is_good_enough(slim_data.V_o, boundary_indices_, boundary_constraints, error, slim_params["tol"]);
				smooth_field = slim_data.V_o.block(0, 0, smooth_field.rows(), dim);
				it += slim_params["min_iter"].get<int>();
			} while (it < max_it && !good_enough);

			for (unsigned i = 0; i < boundary_indices_.rows(); i++)
				smooth_field.row(boundary_indices_(i)) = boundary_constraints.row(i);

			logger().debug("SLIM finished in {} iterations", it);

			if (!good_enough)
				logger().warn("Slimflator could not inflate correctly.");

			return good_enough;
		}
	} // namespace

	ShapeProblem::ShapeProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_, const json &args) : OptimizationProblem(state_, j_, args)
	{
		optimization_name = "shape";
		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		if (args.contains("target_weight"))
			target_weight = args["target_weight"];

		// volume constraint
		volume_weight = args.contains("volume_weight") ? args["volume_weight"].get<double>() : 0;
		target_volume = args.contains("target_volume") ? args["target_volume"].get<double>() : 0;
		if (volume_weight != 0 && !args.contains("target_volume"))
			logger().error("Positive volume constraint weight, but no target volume specified!");
		if (args.contains("penalize_large_volume"))
			penalize_large_volume = args["penalize_large_volume"];

		if (volume_weight > 0)
		{
			j_volume = CompositeFunctional::create("Volume");
			auto &func_volume = *dynamic_cast<VolumeFunctional *>(j_volume.get());
			if (penalize_large_volume)
				func_volume.set_max_volume(target_volume);
			else
				func_volume.set_min_volume(target_volume);
		}
		else
			j_volume = NULL;

		// mesh topology
		Eigen::MatrixXd Vtmp;
		state.get_vf(Vtmp, elements);

		// contact
		if (!opt_params.contains("has_collision"))
			has_collision = state.args["has_collision"];
		else
			has_collision = opt_params["has_collision"];
		if (state.args["has_collision"] && !has_collision)
			logger().warn("Problem has collision, but collision detection in shape optimization is disabled!");
		if (has_collision)
		{
			_dhat = opt_params["dhat"];
			_prev_distance = -1;
			_barrier_stiffness = opt_params["barrier_stiffness"];
			_broad_phase_method = state.args["solver_params"]["broad_phase_method"];
			_ccd_tolerance = state.args["solver_params"]["ccd_tolerance"];
			_ccd_max_iterations = state.args["solver_params"]["ccd_max_iterations"];
		}

		// for higher order bases, need to create a separate collision_mesh for gbases
		if (has_collision && state.bases[0].bases[0].order() > 1)
		{
			logger().error("Contact barrier in shape optimization only supports linear bases!");
			exit(0);
		}

		build_fixed_nodes();

		// smooth constraint
		smoothing_weight = args.contains("smoothing_weight") ? args["smoothing_weight"].get<double>() : 1.;
		adjust_smooth_period = args.contains("adjust_smooth_period") ? args["adjust_smooth_period"].get<int>() : (int)1e9;
		smooth_ratio = args.contains("smooth_ratio") ? args["smooth_ratio"].get<double>() : 1.;
		boundary_smoother.dim = dim;
		if (opt_params.contains("use_weighted_smoothing") && opt_params["use_weighted_smoothing"].get<bool>() && opt_params.contains("weighted_smoothing_power"))
			boundary_smoother.p = opt_params["weighted_smoothing_power"].get<int>();

		Eigen::MatrixXd boundary_nodes_pos;
		Eigen::MatrixXi boundary_edges, boundary_triangles;
		state.extract_boundary_mesh(state.n_geom_bases, gbases, boundary_nodes_pos, boundary_edges, boundary_triangles);
		boundary_smoother.build_laplacian(state.n_geom_bases, state.mesh->dimension(), boundary_edges, state.boundary_gnodes, fixed_nodes);

		// constraints on optimization
		x_to_param = [this](const TVector &x, const Eigen::MatrixXd &V_prev, Eigen::MatrixXd &V) {
			V.setZero(x.size() / this->dim, this->dim);
			for (int i = 0; i < V.rows(); i++)
				for (int d = 0; d < this->dim; d++)
					V(i, d) = x(i * this->dim + d);
		};
		param_to_x = [this](TVector &x, const Eigen::MatrixXd &V) {
			x.setZero(V.rows() * this->dim, 1);
			for (int i = 0; i < V.rows(); i++)
				for (int d = 0; d < this->dim; d++)
					x(i * this->dim + d) = V(i, d);
		};
		dparam_to_dx = [this](TVector &grad_x, const TVector &grad_v) {
			grad_x = grad_v;
			for (int b : this->fixed_nodes)
				for (int d = 0; d < this->dim; d++)
					grad_x(b * this->dim + d) = 0;
		};
	}

	void ShapeProblem::update_constraint_set(const Eigen::MatrixXd &displaced_surface)
	{
		// Store the previous value used to compute the constraint set to avoid
		// duplicate computation.
		static Eigen::MatrixXd cached_displaced_surface;
		if (cached_displaced_surface.size() == displaced_surface.size()
			&& cached_displaced_surface == displaced_surface)
			return;

		if (_use_cached_candidates)
			ipc::construct_constraint_set(
				_candidates, state.collision_mesh, displaced_surface, _dhat,
				_constraint_set);
		else
			ipc::construct_constraint_set(
				state.collision_mesh, displaced_surface, _dhat,
				_constraint_set, /*dmin=*/0, _broad_phase_method);
		cached_displaced_surface = displaced_surface;
	}

	bool ShapeProblem::is_intersection_free(const TVector &x)
	{
		if (!has_collision)
			return true;

		Eigen::MatrixXd V;
		x_to_param(x, state.collision_mesh.vertices_at_rest(), V);

		return !ipc::has_intersections(state.collision_mesh, state.collision_mesh.vertices(V));
	}

	bool ShapeProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
	{
		if (!has_collision)
			return true;

		Eigen::MatrixXd V0, V1;
		x_to_param(x0, state.collision_mesh.vertices_at_rest(), V0);
		x_to_param(x1, state.collision_mesh.vertices_at_rest(), V1);

		// Skip CCD if the displacement is zero.
		if ((V1 - V0).lpNorm<Eigen::Infinity>() == 0.0)
		{
			// Assumes initially intersection-free
			assert(is_intersection_free(x0));
			return true;
		}

		bool is_valid;
		if (_use_cached_candidates)
			is_valid = ipc::is_step_collision_free(
				_candidates, state.collision_mesh,
				state.collision_mesh.vertices(V0),
				state.collision_mesh.vertices(V1),
				_ccd_tolerance, _ccd_max_iterations);
		else
			is_valid = ipc::is_step_collision_free(
				state.collision_mesh,
				state.collision_mesh.vertices(V0),
				state.collision_mesh.vertices(V1),
				ipc::BroadPhaseMethod::HASH_GRID, _ccd_tolerance, _ccd_max_iterations);

		return is_valid;
	}

	double ShapeProblem::target_value(const TVector &x)
	{
		if (mesh_flipped)
			return std::nan("");
		return target_weight * j->energy(state);
	}

	double ShapeProblem::volume_value(const TVector &x)
	{
		if (j_volume && !mesh_flipped)
			return j_volume->energy(state) * volume_weight;
		else
			return 0.;
	}

	double ShapeProblem::smooth_value(const TVector &x)
	{
		Eigen::MatrixXd V;
		x_to_param(x, state.collision_mesh.vertices_at_rest(), V);

		if (opt_params.contains("use_weighted_smoothing") && opt_params["use_weighted_smoothing"].get<bool>())
			return boundary_smoother.weighted_smoothing_energy(V) * smoothing_weight;
		else
			return boundary_smoother.smoothing_energy(V) * smoothing_weight;
	}

	double ShapeProblem::value(const TVector &x)
	{
		if (std::isnan(cur_val))
		{
			double target_val, volume_val, smooth_val, barrier_val;
			target_val = target_value(x);
			volume_val = volume_value(x);
			smooth_val = smooth_value(x);
			barrier_val = barrier_energy(x);
			logger().debug("target = {}, vol = {}, smooth = {}, barrier = {}", target_val, volume_val, smooth_val, barrier_val);
			cur_val = target_val + volume_val + smooth_val + barrier_val;
		}
		return cur_val;
	}

	void ShapeProblem::target_gradient(const TVector &x, TVector &gradv)
	{
		if (mesh_flipped)
		{
			gradv.setZero(x.size());
			return;
		}
		dparam_to_dx(gradv, j->gradient(state, "shape") * target_weight);
	}

	void ShapeProblem::smooth_gradient(const TVector &x, TVector &gradv)
	{
		if (mesh_flipped)
		{
			gradv.setZero(x.size());
			return;
		}
		Eigen::MatrixXd V;
		x_to_param(x, state.collision_mesh.vertices_at_rest(), V);
		TVector grad;
		if (opt_params.contains("use_weighted_smoothing") && opt_params["use_weighted_smoothing"].get<bool>())
			boundary_smoother.weighted_smoothing_grad(V, grad);
		else
			boundary_smoother.smoothing_grad(V, grad);
		grad *= smoothing_weight;

		dparam_to_dx(gradv, grad);
	}

	void ShapeProblem::volume_gradient(const TVector &x, TVector &gradv)
	{
		gradv.setZero(x.size());
		if (!j_volume || mesh_flipped)
			return;

		dparam_to_dx(gradv, j_volume->gradient(state, "shape") * volume_weight);
	}

	double ShapeProblem::barrier_energy(const TVector &x)
	{
		if (!has_collision || mesh_flipped)
			return 0;

		Eigen::MatrixXd V;
		x_to_param(x, state.collision_mesh.vertices_at_rest(), V);

		return _barrier_stiffness * ipc::compute_barrier_potential(state.collision_mesh, state.collision_mesh.vertices(V), _constraint_set, _dhat);
	}

	void ShapeProblem::barrier_gradient(const TVector &x, TVector &gradv)
	{
		if (!has_collision || mesh_flipped)
		{
			gradv.setZero(x.size());
			return;
		}

		Eigen::MatrixXd V;
		x_to_param(x, state.collision_mesh.vertices_at_rest(), V);

		Eigen::MatrixXd displaced_surface = state.collision_mesh.vertices(V);
		TVector grad;
		grad = ipc::compute_barrier_potential_gradient(
			state.collision_mesh, displaced_surface, _constraint_set, _dhat);
		grad = state.collision_mesh.to_full_dof(grad);

		grad *= _barrier_stiffness;

		dparam_to_dx(gradv, grad);
	}

	void ShapeProblem::gradient(const TVector &x, TVector &gradv)
	{
		if (cur_grad.size() == 0)
		{
			Eigen::VectorXd grad_target, grad_smoothing, grad_volume, grad_barrier;
			target_gradient(x, grad_target);
			smooth_gradient(x, grad_smoothing);
			volume_gradient(x, grad_volume);
			barrier_gradient(x, grad_barrier);
			logger().debug("‖∇ target‖ = {}, ‖∇ vol‖ = {}, ‖∇ smooth‖ = {}, ‖∇ barrier‖ = {}", grad_target.norm(), grad_volume.norm(), grad_smoothing.norm(), grad_barrier.norm());
			cur_grad = grad_target + grad_volume + grad_smoothing + grad_barrier;
		}

		gradv = cur_grad;
	}

	void ShapeProblem::smoothing(const TVector &x, TVector &new_x)
	{
		if (opt_params.contains("skip_slim") && opt_params["skip_slim"])
			return;

		Eigen::MatrixXd V, new_V;
		x_to_param(x, state.collision_mesh.vertices_at_rest(), V);

		double rate = 2.;
		bool good_enough = false;
		Eigen::MatrixXd boundary_constraints = Eigen::MatrixXd::Zero(state.boundary_gnodes.size(), dim);

		do
		{
			rate /= 2;
			logger().trace("Try SLIM with step size {}", rate);
			TVector tmp_x = (1. - rate) * x + rate * new_x;
			for (int b = 0; b < state.boundary_gnodes.size(); ++b)
				boundary_constraints.row(b) = tmp_x.block(state.boundary_gnodes[b] * dim, 0, dim, 1).transpose();

			good_enough = internal_smoothing(V, elements, state.boundary_gnodes, boundary_constraints, opt_params["slim"], new_V);
		} while (!good_enough || is_flipped(new_V, elements));

		logger().debug("SLIM succeeds with step size {}", rate);

		param_to_x(new_x, new_V);
		solution_changed(new_x);
	}

	bool ShapeProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		Eigen::MatrixXd V1;
		x_to_param(x1, state.collision_mesh.vertices_at_rest(), V1);
		if (is_flipped(V1, elements))
			return false;

		if (opt_params.contains("max_change"))
		{
			Eigen::MatrixXd V0;
			x_to_param(x0, state.collision_mesh.vertices_at_rest(), V0);
			double change = (V1 - V0).rowwise().norm().maxCoeff();
			if (change > opt_params["max_change"].get<double>())
				return false;
		}

		return true;
	}

	double ShapeProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		if (!has_collision)
			return 1;

		Eigen::MatrixXd V0, V1;
		x_to_param(x0, state.collision_mesh.vertices_at_rest(), V0);
		x_to_param(x1, state.collision_mesh.vertices_at_rest(), V1);

		double max_step = 1;
		assert(!is_flipped(V0, elements));
		while (is_flipped(V0 + max_step * (V1 - V0), elements))
			max_step /= 2.;

		// Extract surface only
		V0 = state.collision_mesh.vertices(V0);
		V1 = state.collision_mesh.vertices(V1);

		auto Vmid = V0 + max_step * (V1 - V0);
		if (_use_cached_candidates)
			max_step *= ipc::compute_collision_free_stepsize(
				_candidates, state.collision_mesh, V0, Vmid,
				_ccd_tolerance, _ccd_max_iterations);
		else
			max_step *= ipc::compute_collision_free_stepsize(
				state.collision_mesh, V0, Vmid,
				_broad_phase_method, _ccd_tolerance, _ccd_max_iterations);
		// polyfem::logger().trace("best step {}", max_step);

		return max_step;
	}

	double ShapeProblem::heuristic_max_step(const TVector &dx)
	{
		return opt_params.contains("max_step") ? opt_params["max_step"].get<double>() : 1;
	}

	void ShapeProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		OptimizationProblem::line_search_begin(x0, x1);
		state.descent_direction = descent_direction;

		x_at_ls_begin = x0;
		sol_at_ls_begin = state.sol;

		if (!has_collision)
			return;

		Eigen::MatrixXd V0, V1;
		x_to_param(x0, state.collision_mesh.vertices_at_rest(), V0);
		x_to_param(x1, state.collision_mesh.vertices_at_rest(), V1);

		ipc::construct_collision_candidates(
			state.collision_mesh,
			state.collision_mesh.vertices(V0),
			state.collision_mesh.vertices(V1),
			_candidates,
			/*inflation_radius=*/_dhat / 1.99, // divide by 1.99 instead of 2 to be conservative
			_broad_phase_method);

		_use_cached_candidates = true;
	}

	void ShapeProblem::line_search_end(bool failed)
	{
		_candidates.clear();
		_use_cached_candidates = false;

		if (opt_params.contains("export_energies"))
		{
			std::ofstream outfile;
			outfile.open(opt_params["export_energies"], std::ofstream::out | std::ofstream::app);

			outfile << value(cur_x) << ", " << target_value(cur_x) << ", " << smooth_value(cur_x) << ", " << volume_value(cur_x) << ", " << barrier_energy(cur_x) << "\n";
			outfile.close();
		}
	}

	void ShapeProblem::save_to_file(const TVector &x0)
	{
		OptimizationProblem::save_to_file(x0);
		if (!state.mesh->is_volume())
			state.mesh->save(state.resolve_output_path(fmt::format("opt_{:d}.obj", iter)));
		else
		{
			Eigen::MatrixXd V;
			Eigen::MatrixXi F;
			state.get_vf(V, F);
			igl::writeMESH(state.resolve_output_path(fmt::format("opt_{:d}.mesh", iter)), V, F, Eigen::MatrixXi());
		}
	}

	void ShapeProblem::post_step(const int iter_num, const TVector &x0)
	{
		if (iter % adjust_smooth_period == 0 && iter > 0)
		{
			double target_val, volume_val, barrier_val;
			target_val = target_value(x0);
			volume_val = volume_value(x0);
			barrier_val = barrier_energy(x0);
			smoothing_weight *= smooth_ratio * (barrier_val + target_val + volume_val) / boundary_smoother.boundary_nodes.size();

			logger().info("update smoothing weight to {}", smoothing_weight);
		}

		iter++;

		if (has_collision)
		{
			Eigen::MatrixXd V;
			x_to_param(x0, state.collision_mesh.vertices_at_rest(), V);
			Eigen::MatrixXd displaced_surface = state.collision_mesh.vertices(V);

			const double dist_sqr = ipc::compute_minimum_distance(state.collision_mesh, displaced_surface, _constraint_set);
			polyfem::logger().trace("min_dist {}", sqrt(dist_sqr));

			_prev_distance = dist_sqr;
		}
	}

	void ShapeProblem::solution_changed(const TVector &newX)
	{
		if (cur_x.size() == newX.size() && cur_x == newX)
			return;

		Eigen::MatrixXd V;
		x_to_param(newX, state.collision_mesh.vertices_at_rest(), V);
		auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;
		state.set_v(V);
		mesh_flipped = is_flipped(V, elements);
		if (mesh_flipped)
			logger().debug("Mesh Flipped!");
		else
			solve_pde(newX);

		cur_x = newX;
		cur_grad.resize(0);
		cur_val = std::nan("");

		if (!has_collision || mesh_flipped)
			return;

		update_constraint_set(state.collision_mesh.vertices(V));
	}

	bool ShapeProblem::remesh(TVector &x)
	{
		// quality check
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		state.get_vf(V, F);
		Eigen::VectorXd quality;
		scaled_jacobian(V, F, quality);

		double min_quality = quality.minCoeff();
		double avg_quality = quality.sum() / quality.size();
		logger().debug("Mesh worst quality: {}, avg quality: {}", min_quality, avg_quality);
		if (!opt_params.contains("remesh_quality") || min_quality > opt_params["remesh_quality"].get<double>())
			return false;

		logger().info("Remeshing ...");

		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		// if (state.mesh->is_volume())
		// {
		// }
		// else
		// {
		//     // extract V and F of the whole mesh
		//     V.setZero(state.mesh->n_vertices(), state.mesh->dimension());
		//     F.setZero(state.mesh->n_faces(), state.mesh->dimension() + 1);
		//     for (int v = 0; v < V.rows(); v++)
		//         V.row(v) = state.mesh->point(v);
		// }

		if (V.cols() < 3)
		{
			V.conservativeResize(V.rows(), 3);
			V.col(2).setZero();
		}

		// if (state.mesh->is_volume())
		// {
		//     const Mesh3D &tmp_mesh = *dynamic_cast<Mesh3D *>(state.mesh.get());
		//     for (int f = 0; f < F.rows(); f++)
		//         for (int v = 0; v < F.cols(); v++)
		//             F(f, v) = tmp_mesh.cell_vertex(f, v);
		// }
		// else
		// {
		//     const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(state.mesh.get());
		//     for (int f = 0; f < F.rows(); f++)
		//         for (int v = 0; v < F.cols(); v++)
		//             F(f, v) = tmp_mesh.face_vertex(f, v);
		// }

		if (is_flipped(V, F))
		{
			logger().error("Mesh flippped during remeshing!");
			exit(0);
		}

		if (state.mesh->n_meshes() == 0)
		{
			logger().error("Use meshes instead of mesh when remeshing enabled!");
			return false;
		}

		std::set<int> optimize_body_ids;
		if (opt_params.contains("optimize_body_ids"))
		{
			for (int i : opt_params["optimize_body_ids"])
				optimize_body_ids.insert(i);
		}
		else
		{
			for (int m = 0; m < state.args["meshes"].get<std::vector<json>>().size(); m++)
				optimize_body_ids.insert(state.args["meshes"][m]["body_id"].get<int>());
		}

		{
			for (int body_id : optimize_body_ids)
			{
				// build submesh
				Eigen::VectorXi vertex_mask, cell_mask, vertex_map(V.rows());
				vertex_mask.setZero(V.rows());
				cell_mask.setZero(F.rows());
				vertex_map.setConstant(-1);
				for (int e = 0; e < gbases.size(); e++)
				{
					if (state.mesh->get_body_id(e) != body_id)
						continue;
					cell_mask[e] = 1;
				}
				for (int f = 0; f < F.rows(); f++)
					for (int v = 0; v < F.cols(); v++)
						vertex_mask[F(f, v)] = 1;
				int idx = 0;
				for (int v = 0; v < V.rows(); v++)
					if (vertex_mask[v])
					{
						vertex_map[v] = idx;
						idx++;
					}

				Eigen::MatrixXd Vm(vertex_mask.sum(), 3);
				Eigen::MatrixXi Fm(cell_mask.sum(), state.mesh->dimension() + 1);
				for (int v = 0; v < V.rows(); v++)
					if (vertex_map[v] >= 0)
						Vm.row(vertex_map[v]) = V.row(v);
				idx = 0;
				for (int f = 0; f < F.rows(); f++)
					if (cell_mask[f])
					{
						for (int v = 0; v < F.cols(); v++)
							Fm(idx, v) = vertex_map[F(f, v)];
						idx++;
					}

				std::string before_remesh_path, after_remesh_path;
				if (!state.mesh->is_volume())
				{
					before_remesh_path = state.resolve_output_path(fmt::format("before_remesh_iter{:d}_mesh{:d}.obj", iter, body_id));
					after_remesh_path = state.resolve_output_path(fmt::format("after_remesh_iter{:d}_mesh{:d}.msh", iter, body_id));

					igl::write_triangle_mesh(before_remesh_path, Vm, Fm);

					double target_length = igl::avg_edge_length(Vm, Fm);

					std::string command = "python remesh.py " + before_remesh_path + " " + after_remesh_path;
					int return_val = system(command.c_str());
					if (return_val == 0)
						logger().info("remesh command \"{}\" returns {}", command, return_val);
					else
					{
						logger().error("remesh command \"{}\" returns {}", command, return_val);
						return false;
					}
				}
				else
				{
					before_remesh_path = state.resolve_output_path(fmt::format("before_remesh_iter{:d}_mesh{:d}.mesh", iter, body_id));
					after_remesh_path = state.resolve_output_path(fmt::format("after_remesh_iter{:d}_mesh{:d}.msh", iter, body_id));

					igl::writeMESH(before_remesh_path, Vm, Fm, Eigen::MatrixXi());

					auto tmp_before_remesh_path = StringUtils::replace_ext(before_remesh_path, "msh");

					int return_val = system(("gmsh " + before_remesh_path + " -save -format msh22 -o " + tmp_before_remesh_path).c_str());
					if (return_val != 0)
					{
						logger().error("gmsh command \"{}\" returns {}", "gmsh " + before_remesh_path + " -save -format msh22 -o " + tmp_before_remesh_path, return_val);
						return false;
					}

					if (!opt_params.contains("remesh_exe"))
					{
						logger().error("No remeshing executable specified!");
						return false;
					}
					else
					{
						std::string command = opt_params["remesh_exe"].template get<std::string>() + " " + tmp_before_remesh_path + " " + after_remesh_path + " -j 10";
						return_val = system(command.c_str());
						if (return_val == 0)
							logger().info("remesh command \"{}\" returns {}", command, return_val);
						else
						{
							logger().error("remesh command \"{}\" returns {}", command, return_val);
							return false;
						}
					}
				}

				// modify json
				bool flag = false;
				for (int m = 0; m < state.args["meshes"].get<std::vector<json>>().size(); m++)
				{
					if (state.args["meshes"][m]["body_id"].get<int>() != body_id)
						continue;
					if (!flag)
						flag = true;
					else
					{
						logger().error("Multiple meshes found with same body id!");
						return false;
					}
					state.args["meshes"][m]["skip_transform"] = true;
					state.args["meshes"][m]["mesh"] = after_remesh_path;
				}
			}
		}

		// initialize things after remeshing
		state.mesh.reset();
		state.assembler.update_lame_params(Eigen::MatrixXd(), Eigen::MatrixXd());

		state.load_mesh();
		state.compute_mesh_stats();
		state.build_basis();

		state.sol.resize(0, 0);
		state.descent_direction.resize(0);

		Eigen::MatrixXd Vtmp;
		state.get_vf(Vtmp, elements);
		param_to_x(x, Vtmp);

		build_fixed_nodes();

		cur_grad.resize(0);
		cur_val = std::nan("");

		sol_at_ls_begin.resize(0, 0);
		x_at_ls_begin.resize(0);

		Eigen::MatrixXd boundary_nodes_pos;
		Eigen::MatrixXi boundary_edges, boundary_triangles;
		state.extract_boundary_mesh(state.n_geom_bases, gbases, boundary_nodes_pos, boundary_edges, boundary_triangles);
		boundary_smoother.build_laplacian(state.n_geom_bases, state.mesh->dimension(), boundary_edges, state.boundary_gnodes, fixed_nodes);

		logger().info("Remeshing done!");

		state.get_vf(V, F);
		scaled_jacobian(V, F, quality);
		if (quality.minCoeff() <= opt_params["remesh_quality"].get<double>())
		{
			logger().error("Quality not good after remeshing!");
			exit(0);
		}

		return true;
	}

	void ShapeProblem::build_fixed_nodes()
	{
		fixed_nodes.clear();
		const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

		Eigen::MatrixXd V;
		state.get_vf(V, elements);
		V.conservativeResize(V.rows(), state.mesh->dimension());

		// fix certain object
		std::set<int> optimize_body_ids;
		std::set<int> optimize_boundary_ids;
		if (opt_params.contains("optimize_body_ids"))
		{
			for (int i : opt_params["optimize_body_ids"])
				optimize_body_ids.insert(i);

			for (int e = 0; e < state.bases.size(); e++)
			{
				const int body_id = state.mesh->get_body_id(e);
				if (!optimize_body_ids.count(body_id))
					for (const auto &gbs : gbases[e].bases)
						for (const auto &g : gbs.global())
							fixed_nodes.insert(g.index);
			}
		}
		else if (opt_params.contains("optimize_boundary_ids"))
		{
			for (int i : opt_params["optimize_boundary_ids"])
				optimize_boundary_ids.insert(i);

			for (const auto &lb : state.total_local_boundary)
			{
				const int e = lb.element_id();
				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const auto nodes = state.bases[e].local_nodes_for_primitive(primitive_global_id, *state.mesh);

					if (!optimize_boundary_ids.count(state.mesh->get_boundary_id(primitive_global_id)))
					{
						for (long n = 0; n < nodes.size(); ++n)
						{
							fixed_nodes.insert(state.bases[e].bases[nodes(n)].global()[0].index);
						}
					}
					else
					{
						for (long n = 0; n < nodes.size(); ++n)
						{
							if (optimization_boundary_to_node.find(state.mesh->get_boundary_id(primitive_global_id)) != optimization_boundary_to_node.end())
							{
								optimization_boundary_to_node[state.mesh->get_boundary_id(primitive_global_id)].push_back(state.bases[e].bases[nodes(n)].global()[0].index);
							}
							else
							{
								optimization_boundary_to_node[state.mesh->get_boundary_id(primitive_global_id)] = {state.bases[e].bases[nodes(n)].global()[0].index};
							}
						}
					}
				}
			}
		}
		else
			logger().info("No optimization body or boundary specified, optimize shape of every mesh...");

		// fix dirichlet bc
		if (!opt_params.contains("fix_dirichlet") || opt_params["fix_dirichlet"].get<bool>() == true)
		{
			for (const auto &lb : state.local_boundary)
			{
				for (int i = 0; i < lb.size(); ++i)
				{
					const int e = lb.element_id();
					const int primitive_g_id = lb.global_primitive_id(i);
					const int tag = state.mesh->get_boundary_id(primitive_g_id);
					const auto nodes = gbases[e].local_nodes_for_primitive(primitive_g_id, *(state.mesh));

					if (tag <= 0)
						continue;

					for (long n = 0; n < nodes.size(); ++n)
					{
						assert(gbases[e].bases[nodes(n)].global().size() == 1);
						fixed_nodes.insert(gbases[e].bases[nodes(n)].global()[0].index);
					}
				}
			}
		}

		// fix neumann bc
		for (const auto &lb : state.local_neumann_boundary)
		{
			for (int i = 0; i < lb.size(); ++i)
			{
				const int e = lb.element_id();
				const int primitive_g_id = lb.global_primitive_id(i);
				const int tag = state.mesh->get_boundary_id(primitive_g_id);
				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_g_id, *(state.mesh));

				if (tag <= 0)
					continue;

				for (long n = 0; n < nodes.size(); ++n)
				{
					assert(gbases[e].bases[nodes(n)].global().size() == 1);
					fixed_nodes.insert(gbases[e].bases[nodes(n)].global()[0].index);
				}
			}
		}

		// fix contact area, need threshold
		if (opt_params.contains("fix_contact_area_threshold"))
		{
			assert(state.n_geom_bases == state.n_bases && "Higer order basis not supported, need a separate collision mesh for gbases!");
			const double threshold = opt_params["fix_contact_area_threshold"].get<double>();

			ipc::Constraints contact_set;
			ipc::construct_constraint_set(state.collision_mesh, state.collision_mesh.vertices(V), threshold, contact_set);

			for (int c = 0; c < contact_set.ee_constraints.size(); c++)
			{
				const auto &constraint = contact_set.ee_constraints[c];

				if (constraint.compute_distance(state.collision_mesh.vertices(V), state.collision_mesh.edges(), state.collision_mesh.faces()) >= threshold)
					continue;

				fixed_nodes.insert(state.collision_mesh.to_full_vertex_id(constraint.vertex_indices(state.collision_mesh.edges(), state.collision_mesh.faces())[0]));
				fixed_nodes.insert(state.collision_mesh.to_full_vertex_id(constraint.vertex_indices(state.collision_mesh.edges(), state.collision_mesh.faces())[1]));
			}

			contact_set.ee_constraints.clear();

			for (int c = 0; c < contact_set.size(); c++)
			{
				const auto &constraint = contact_set[c];

				if (constraint.compute_distance(state.collision_mesh.vertices(V), state.collision_mesh.edges(), state.collision_mesh.faces()) >= threshold)
					continue;

				fixed_nodes.insert(state.collision_mesh.to_full_vertex_id(constraint.vertex_indices(state.collision_mesh.edges(), state.collision_mesh.faces())[0]));
			}
		}

		// fix nodes based on problem setting
		if (opt_params.contains("name") && opt_params["name"].get<std::string>() == "door")
		{
			for (int i = 0; i < V.rows(); i++)
			{
				// the boundary that touches the door
				// if (V(i, 0) > -0.883353 && V(i, 0) < 0.177471 && V(i, 1) < 0.824529)
				// 	fixed_nodes.insert(i);
				// the handle of hook
				if (V(i, 0) > 0.7647)
					fixed_nodes.insert(i);
				// the left end of hook, to not let it degenerate
				// else if (V(i, 1) < 0.471 && V(i, 0) < -0.88 && V(i, 0) > -0.95)
				// 	fixed_nodes.insert(i);
			}
		}

		logger().info("Fixed nodes: {}", fixed_nodes.size());
	}
} // namespace polyfem

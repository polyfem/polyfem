#include "ShapeParameter.hpp"

#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <igl/slim.h>
#include <igl/writeOBJ.h>
#include <igl/writeMESH.h>
#include <igl/write_triangle_mesh.h>
#include <igl/avg_edge_length.h>

#include <ipc/ipc.hpp>
#include <ipc/barrier/barrier.hpp>
#include <ipc/barrier/adaptive_stiffness.hpp>
#include <ipc/utils/world_bbox_diagonal_length.hpp>
#include <polyfem/utils/BoundarySampler.hpp>

#include <filesystem>

namespace ipc
{
	NLOHMANN_JSON_SERIALIZE_ENUM(
		ipc::BroadPhaseMethod,
		{{ipc::BroadPhaseMethod::HASH_GRID, "hash_grid"}, // also default
		 {ipc::BroadPhaseMethod::HASH_GRID, "HG"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "brute_force"},
		 {ipc::BroadPhaseMethod::BRUTE_FORCE, "BF"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "spatial_hash"},
		 {ipc::BroadPhaseMethod::SPATIAL_HASH, "SH"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "sweep_and_tiniest_queue"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE, "STQ"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "sweep_and_tiniest_queue_gpu"},
		 {ipc::BroadPhaseMethod::SWEEP_AND_TINIEST_QUEUE_GPU, "STQ_GPU"}})
} // namespace ipc

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

			quality.setZero(F.rows());
			if (dim == 2)
			{
				for (int i = 0; i < F.rows(); i++)
				{
					Eigen::RowVector3d e0;
					e0(2) = 0;
					e0.head(2) = V.row(F(i, 2)) - V.row(F(i, 1));
					Eigen::RowVector3d e1;
					e1(2) = 0;
					e1.head(2) = V.row(F(i, 0)) - V.row(F(i, 2));
					Eigen::RowVector3d e2;
					e2(2) = 0;
					e2.head(2) = V.row(F(i, 1)) - V.row(F(i, 0));

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

	ShapeParameter::ShapeParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args)
		: Parameter(states_ptr, args)
	{
		parameter_name_ = "shape";

		dim = states_ptr_[0]->mesh->dimension();

		full_dim_ = states_ptr_[0]->n_geom_bases * dim;

		const auto &gbases = states_ptr_[0]->geom_bases();

		// mesh topology
		states_ptr_[0]->get_vf(V_rest, elements);

		// contact
		const auto &opt_contact_params = args["contact"];
		has_collision = opt_contact_params["enabled"];
		if (states_ptr_[0]->is_contact_enabled() && !has_collision)
			logger().warn("Problem has collision, but collision detection in shape optimization is disabled!");
		if (has_collision)
		{
			_broad_phase_method = opt_contact_params["CCD"]["broad_phase"];
			_ccd_tolerance = opt_contact_params["CCD"]["tolerance"];
			_ccd_max_iterations = opt_contact_params["CCD"]["max_iterations"];
		}

		Eigen::MatrixXd boundary_nodes_pos;
		states_ptr_[0]->build_collision_mesh(boundary_nodes_pos, collision_mesh, states_ptr_[0]->n_geom_bases, gbases);

		shape_params = args;
		if (shape_params.contains("smoothing_paramters"))
			slim_params = shape_params["smoothing_paramters"];
		// TODO: put me in json spec
		if (slim_params.empty())
		{
			slim_params = json::parse(R"(
			{
				"min_iter" : 2,
				"tol" : 1e-8,
				"soft_p" : 1e5,
				"exp_factor" : 5,
				"skip": false
			}
			)");
		}

		build_active_nodes();
		build_tied_nodes();

		if (shape_params["dimensions"].is_array())
			free_dimension = shape_params["dimensions"].get<std::vector<bool>>();
		if (free_dimension.size() != dim)
			free_dimension.resize(dim, true);

		shape_constraints_ = std::make_unique<ShapeConstraints>(args, V_rest, optimization_boundary_to_node);
		shape_constraints_->set_active_nodes_mask(active_nodes_mask);
		optimization_dim_ = shape_constraints_->get_optimization_dim();
	}

	Eigen::MatrixXd ShapeParameter::map(const Eigen::VectorXd &x) const
	{
		Eigen::MatrixXd V_full;
		shape_constraints_->reduced_to_full(x, V_rest, V_full);
		return V_full;
	}
	Eigen::VectorXd ShapeParameter::map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const
	{
		Eigen::VectorXd dreduced;
		shape_constraints_->dfull_to_dreduced(x, full_grad, dreduced);
		return dreduced;
	}

	bool ShapeParameter::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		if (!has_collision)
			return true;

		Eigen::MatrixXd V0, V1;
		shape_constraints_->reduced_to_full(x0, V_rest, V0);
		shape_constraints_->reduced_to_full(x1, V_rest, V1);

		// Skip CCD if the displacement is zero.
		if ((V1 - V0).lpNorm<Eigen::Infinity>() == 0.0)
			return true;

		bool is_valid;
		is_valid = ipc::is_step_collision_free(
			collision_mesh,
			collision_mesh.vertices(V0),
			collision_mesh.vertices(V1),
			_broad_phase_method, _ccd_tolerance, _ccd_max_iterations);

		return is_valid;
	}

	bool ShapeParameter::smoothing(const Eigen::VectorXd &x, Eigen::VectorXd &new_x)
	{
		if (slim_params["skip"])
		{
			new_x = x;
			return false;
		}

		Eigen::MatrixXd V, new_V;
		shape_constraints_->reduced_to_full(x, V_rest, V);

		double rate = 2.;
		bool good_enough = false;
		Eigen::MatrixXd boundary_constraints = Eigen::MatrixXd::Zero(states_ptr_[0]->boundary_gnodes.size(), dim);

		do
		{
			rate /= 2;
			logger().trace("Try SLIM with step size {}", rate);
			Eigen::VectorXd tmp_x = (1. - rate) * x + rate * new_x;
			Eigen::MatrixXd tmp_V;
			shape_constraints_->reduced_to_full(tmp_x, V_rest, tmp_V);
			for (int b = 0; b < states_ptr_[0]->boundary_gnodes.size(); ++b)
				boundary_constraints.row(b) = tmp_V.block(states_ptr_[0]->boundary_gnodes[b], 0, 1, dim);

			good_enough = internal_smoothing(V, elements, states_ptr_[0]->boundary_gnodes, boundary_constraints, slim_params, new_V);
		} while (!good_enough || is_flipped(new_V, elements));

		logger().debug("SLIM succeeds with step size {}", rate);

		V_rest = new_V;
		shape_constraints_->full_to_reduced(new_V, new_x);
		pre_solve(new_x);

		return true;
	}

	bool ShapeParameter::is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		Eigen::MatrixXd V1;
		shape_constraints_->reduced_to_full(x1, V_rest, V1);
		if (is_flipped(V1, elements))
			return false;

		return true;
	}

	double ShapeParameter::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
		Eigen::MatrixXd V0, V1;
		shape_constraints_->reduced_to_full(x0, V_rest, V0);
		shape_constraints_->reduced_to_full(x1, V_rest, V1);

		double max_step = 1;
		assert(!is_flipped(V0, elements));
		while (is_flipped(V0 + max_step * (V1 - V0), elements))
			max_step /= 2.;

		if (!has_collision)
			return max_step;

		// Extract surface only
		V0 = collision_mesh.vertices(V0);
		V1 = collision_mesh.vertices(V1);

		auto Vmid = V0 + max_step * (V1 - V0);
		max_step *= ipc::compute_collision_free_stepsize(
			collision_mesh, V0, Vmid,
			_broad_phase_method, _ccd_tolerance, _ccd_max_iterations);
		// polyfem::logger().trace("best step {}", max_step);

		return max_step;
	}

	void ShapeParameter::line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1)
	{
	}

	void ShapeParameter::line_search_end()
	{
	}

	void ShapeParameter::post_step(const int iter_num, const Eigen::VectorXd &x0)
	{
		iter++;
	}

	bool ShapeParameter::pre_solve(const Eigen::VectorXd &newX)
	{
		Eigen::MatrixXd V;
		shape_constraints_->reduced_to_full(newX, V_rest, V);
		mesh_flipped = is_flipped(V, elements);
		if (mesh_flipped)
		{
			logger().debug("Mesh Flipped!");
			return false;
		}

		for (auto state : states_ptr_)
		{
			state->set_mesh_vertices(V);
			state->build_basis();
		}
		return true;
	}

	void ShapeParameter::post_solve(const Eigen::VectorXd &newX)
	{
	}

	bool ShapeParameter::remesh(Eigen::VectorXd &x)
	{
		// quality check
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		states_ptr_[0]->get_vf(V, F);
		Eigen::VectorXd quality;
		scaled_jacobian(V, F, quality);

		double min_quality = quality.minCoeff();
		double avg_quality = quality.sum() / quality.size();
		logger().debug("Mesh worst quality: {}, avg quality: {}", min_quality, avg_quality);

		static int no_remesh_iter = 1;

		bool should_remesh = false;

		json remesh_args = shape_params["remesh"];
		if (remesh_args.size() == 0)
			return false;

		if (min_quality < remesh_args["tolerance"].get<double>())
		{
			should_remesh = true;
			logger().debug("Remesh due to bad quality...");
		}

		if (no_remesh_iter % remesh_args["period"].get<int>() == 0)
		{
			should_remesh = true;
			logger().debug("Remesh every {} iter...", remesh_args["period"].get<int>());
		}

		if (!should_remesh)
		{
			no_remesh_iter++;
			return false;
		}

		no_remesh_iter = 1;

		// logger().info("Remeshing ...");

		const auto &gbases = states_ptr_[0]->geom_bases();

		if (V.cols() < 3)
		{
			V.conservativeResize(V.rows(), 3);
			V.col(2).setZero();
		}

		if (is_flipped(V, F))
		{
			logger().error("Mesh flippped during remeshing!");
			exit(0);
		}

		std::set<int> optimize_body_ids;
		if (shape_params.contains("volume_selection"))
		{
			for (int i : shape_params["volume_selection"])
				optimize_body_ids.insert(i);
		}
		else
		{
			for (auto &geometry : states_ptr_[0]->args["geometry"])
			{
				if (geometry["volume_selection"].is_number_integer())
					optimize_body_ids.insert(geometry["volume_selection"].get<int>());
				else
					logger().error("Remeshing doesn't support single geometry with multiply volume selections!");
			}
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
					if (states_ptr_[0]->mesh->get_body_id(e) != body_id)
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
				Eigen::MatrixXi Fm(cell_mask.sum(), states_ptr_[0]->mesh->dimension() + 1);
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
				if (!states_ptr_[0]->mesh->is_volume())
				{
					before_remesh_path = states_ptr_[0]->resolve_output_path(fmt::format("before_remesh_iter{:d}_mesh{:d}.obj", iter, body_id));
					after_remesh_path = states_ptr_[0]->resolve_output_path(fmt::format("after_remesh_iter{:d}_mesh{:d}.msh", iter, body_id));

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
					before_remesh_path = states_ptr_[0]->resolve_output_path(fmt::format("before_remesh_iter{:d}_mesh{:d}.mesh", iter, body_id));
					after_remesh_path = states_ptr_[0]->resolve_output_path(fmt::format("after_remesh_iter{:d}_mesh{:d}.msh", iter, body_id));

					igl::writeMESH(before_remesh_path, Vm, Fm, Eigen::MatrixXi());

					auto tmp_before_remesh_path = utils::StringUtils::replace_ext(before_remesh_path, "msh");

					int return_val = system(("gmsh " + before_remesh_path + " -save -format msh22 -o " + tmp_before_remesh_path).c_str());
					if (return_val != 0)
					{
						logger().error("gmsh command \"{}\" returns {}", "gmsh " + before_remesh_path + " -save -format msh22 -o " + tmp_before_remesh_path, return_val);
						return false;
					}

					{
						std::string command = remesh_args["remesh_exe"].template get<std::string>() + " " + tmp_before_remesh_path + " " + after_remesh_path + " -j 10";
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
				for (auto state : states_ptr_)
				{
					bool flag = false;
					for (int m = 0; m < state->in_args["geometry"].get<std::vector<json>>().size(); m++)
					{
						if (state->in_args["geometry"][m]["volume_selection"].get<int>() != body_id)
							continue;
						if (!flag)
							flag = true;
						else
						{
							logger().error("Multiple meshes found with same body id!");
							return false;
						}
						state->in_args["geometry"][m]["transformation"]["skip"] = true;
						state->in_args["geometry"][m]["mesh"] = after_remesh_path;
					}
				}
			}
		}

		// initialize things after remeshing
		for (auto state : states_ptr_)
		{
			state->mesh.reset();
			state->mesh = nullptr;
			state->assembler.update_lame_params(Eigen::MatrixXd(), Eigen::MatrixXd());

			json in_args = state->in_args;
			for (auto &geo : in_args["geometry"])
				if (geo.contains("transformation"))
					geo.erase("transformation");
			std::cout << in_args << std::endl;
			state->init(in_args, false);

			state->load_mesh();
			state->stats.compute_mesh_stats(*state->mesh);
			state->build_basis();
		}
		states_ptr_[0]->get_vf(V_rest, elements);
		shape_constraints_->full_to_reduced(V_rest, x);

		Eigen::MatrixXd boundary_nodes_pos;
		states_ptr_[0]->build_collision_mesh(boundary_nodes_pos, collision_mesh, states_ptr_[0]->n_geom_bases, gbases);

		build_active_nodes();
		build_tied_nodes();

		logger().info("Remeshing finished!");

		states_ptr_[0]->get_vf(V, F);
		scaled_jacobian(V, F, quality);

		min_quality = quality.minCoeff();
		avg_quality = quality.sum() / quality.size();
		logger().debug("Mesh worst quality: {}, avg quality: {}", min_quality, avg_quality);

		return true;
	}

	void ShapeParameter::build_tied_nodes()
	{
		const double correspondence_threshold = shape_params.value("correspondence_threshold", 1e-8);
		const double displace_dist = shape_params.value("displace_dist", 1e-4);

		tied_nodes.clear();
		tied_nodes_mask.assign(V_rest.rows(), false);
		for (int i = 0; i < V_rest.rows(); i++)
		{
			for (int j = 0; j < i; j++)
			{
				if ((V_rest.row(i) - V_rest.row(j)).norm() < correspondence_threshold)
				{
					tied_nodes.push_back(std::array<int, 2>({{i, j}}));
					tied_nodes_mask[i] = true;
					tied_nodes_mask[j] = true;
					logger().trace("Tie {} and {}", i, j);
					break;
				}
			}
		}

		if (tied_nodes.size() == 0)
			return;

		assembler::ElementAssemblyValues vals;
		Eigen::MatrixXd uv, samples, gtmp, rhs_fun;
		Eigen::VectorXi global_primitive_ids;
		Eigen::MatrixXd points, normals;
		Eigen::VectorXd weights;
		const auto &gbases = states_ptr_[0]->geom_bases();

		Eigen::VectorXd vertex_perturbation;
		vertex_perturbation.setZero(states_ptr_[0]->n_geom_bases * dim, 1);

		Eigen::VectorXi n_shared_edges;
		n_shared_edges.setZero(states_ptr_[0]->n_geom_bases);
		for (const auto &lb : states_ptr_[0]->total_local_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, 1, *states_ptr_[0]->mesh, false, uv, points, normals, weights, global_primitive_ids);

			if (!has_samples)
				continue;

			const basis::ElementBases &gbs = gbases[e];

			vals.compute(e, states_ptr_[0]->mesh->is_volume(), points, gbs, gbs);

			const int n_quad_pts = weights.size() / lb.size();
			for (int n = 0; n < vals.jac_it.size(); ++n)
			{
				normals.row(n) = normals.row(n) * vals.jac_it[n];
				normals.row(n).normalize();
			}

			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *states_ptr_[0]->mesh);

				for (long n = 0; n < nodes.size(); ++n)
				{
					const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];
					assert(v.global.size() == 1);

					if (tied_nodes_mask[v.global[0].index])
					{
						vertex_perturbation(Eigen::seqN(v.global[0].index * dim, dim)) -= normals(n_quad_pts * i, Eigen::seqN(0, dim)).transpose();
						n_shared_edges(v.global[0].index) += 1;
					}
				}
			}
		}
		for (int i = 0; i < n_shared_edges.size(); i++)
			if (n_shared_edges(i) > 1)
				vertex_perturbation(Eigen::seqN(i * dim, dim)) *= displace_dist / vertex_perturbation(Eigen::seqN(i * dim, dim)).norm();
		states_ptr_[0]->pre_sol = states_ptr_[0]->down_sampling_mat.transpose() * vertex_perturbation;
	}

	void ShapeParameter::build_active_nodes()
	{
		const auto &mesh = get_state().mesh;
		const auto &bases = get_state().bases;
		const auto &gbases = get_state().geom_bases();

		Eigen::MatrixXd V;
		get_state().get_vf(V, elements);

		active_nodes_mask.assign(get_state().n_geom_bases, true);
		for (const auto &lb : get_state().total_local_boundary)
		{
			const int e = lb.element_id();
			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

				for (long n = 0; n < nodes.size(); ++n)
				{
					const int g_id = gbases[e].bases[nodes(n)].global()[0].index;
					active_nodes_mask[g_id] = false;
				}
			}
		}

		// select certain object
		std::set<int> optimize_body_ids;
		std::set<int> optimize_boundary_ids;
		if (shape_params["volume_selection"].size() > 0)
		{
			for (int i : shape_params["volume_selection"])
				optimize_body_ids.insert(i);

			logger().debug("Optimize shape based on volume selection...");

			for (int e = 0; e < gbases.size(); e++)
			{
				const int body_id = mesh->get_body_id(e);
				if (optimize_body_ids.count(body_id))
					for (const auto &gbs : gbases[e].bases)
						for (const auto &g : gbs.global())
							active_nodes_mask[g.index] = true;
			}
		}
		else if (shape_params["surface_selection"].size() > 0)
		{
			for (int i : shape_params["surface_selection"])
				optimize_boundary_ids.insert(i);

			logger().debug("Optimize shape based on surface selection...");

			for (const auto &lb : get_state().total_local_boundary)
			{
				const int e = lb.element_id();
				for (int i = 0; i < lb.size(); ++i)
				{
					const int primitive_global_id = lb.global_primitive_id(i);
					const int boundary_id = mesh->get_boundary_id(primitive_global_id);
					const auto nodes = gbases[e].local_nodes_for_primitive(primitive_global_id, *mesh);

					if (optimize_boundary_ids.count(boundary_id))
					{
						for (long n = 0; n < nodes.size(); ++n)
						{
							const int g_id = gbases[e].bases[nodes(n)].global()[0].index;
							active_nodes_mask[g_id] = true;

							if (optimization_boundary_to_node.count(boundary_id) != 0)
							{
								if (std::count(optimization_boundary_to_node[boundary_id].begin(), optimization_boundary_to_node[boundary_id].end(), g_id) == 0)
								{
									optimization_boundary_to_node[boundary_id].push_back(g_id);
								}
							}
							else
							{
								optimization_boundary_to_node[boundary_id] = {g_id};
							}
						}
					}
				}
			}
		}
		else
		{
			logger().debug("No volume or surface selection specified, optimize shape of every mesh...");

			active_nodes_mask.assign(active_nodes_mask.size(), true);
		}
		// fix dirichlet bc
		if (!shape_params.contains("fix_dirichlet") || shape_params["fix_dirichlet"].get<bool>())
		{
			logger().info("Fix position of Dirichlet boundary nodes.");
			for (const auto &lb : get_state().local_boundary)
			{
				for (int i = 0; i < lb.size(); ++i)
				{
					const int e = lb.element_id();
					const int primitive_g_id = lb.global_primitive_id(i);
					const auto nodes = gbases[e].local_nodes_for_primitive(primitive_g_id, *(get_state().mesh));

					for (long n = 0; n < nodes.size(); ++n)
					{
						assert(gbases[e].bases[nodes(n)].global().size() == 1);
						active_nodes_mask[gbases[e].bases[nodes(n)].global()[0].index] = false;
					}
				}
			}
		}

		// fix neumann bc
		logger().info("Fix position of nonzero Neumann boundary nodes.");
		for (const auto &lb : get_state().local_neumann_boundary)
		{
			for (int i = 0; i < lb.size(); ++i)
			{
				const int e = lb.element_id();
				const int primitive_g_id = lb.global_primitive_id(i);
				const auto nodes = gbases[e].local_nodes_for_primitive(primitive_g_id, *(get_state().mesh));

				for (long n = 0; n < nodes.size(); ++n)
				{
					assert(gbases[e].bases[nodes(n)].global().size() == 1);
					active_nodes_mask[gbases[e].bases[nodes(n)].global()[0].index] = false;
				}
			}
		}

		// fix contact area, need threshold
		if (shape_params.contains("fix_contact_surface") && shape_params["fix_contact_surface"].get<bool>())
		{
			const double threshold = shape_params["fix_contact_surface_tol"].get<double>();
			logger().info("Fix position of boundary nodes in contact.");

			ipc::Constraints contact_set;
			contact_set.build(collision_mesh, collision_mesh.vertices(V), threshold);

			for (int c = 0; c < contact_set.ee_constraints.size(); c++)
			{
				const auto &constraint = contact_set.ee_constraints[c];

				if (constraint.compute_distance(collision_mesh.vertices(V), collision_mesh.edges(), collision_mesh.faces()) >= threshold)
					continue;

				active_nodes_mask[collision_mesh.to_full_vertex_id(constraint.vertex_indices(collision_mesh.edges(), collision_mesh.faces())[0])] = false;
				active_nodes_mask[collision_mesh.to_full_vertex_id(constraint.vertex_indices(collision_mesh.edges(), collision_mesh.faces())[1])] = false;
			}

			contact_set.ee_constraints.clear();

			for (int c = 0; c < contact_set.size(); c++)
			{
				const auto &constraint = contact_set[c];

				if (constraint.compute_distance(collision_mesh.vertices(V), collision_mesh.edges(), collision_mesh.faces()) >= threshold)
					continue;

				active_nodes_mask[collision_mesh.to_full_vertex_id(constraint.vertex_indices(collision_mesh.edges(), collision_mesh.faces())[0])] = false;
			}
		}

		int num = 0;
		for (bool b : active_nodes_mask)
			if (!b)
				num++;

		logger().info("Fixed nodes in shape optimization: {}", num);
	}
} // namespace polyfem

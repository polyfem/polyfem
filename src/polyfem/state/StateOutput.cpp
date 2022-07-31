#include <polyfem/State.hpp>
#include <polyfem/utils/par_for.hpp>

#include <polyfem/utils/BoundarySampler.hpp>

#include <polyfem/mesh/VTUWriter.hpp>
#include <polyfem/mesh/MeshUtils.hpp>

#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/ALNLProblem.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <polyfem/utils/Timer.hpp>

// #ifdef POLYFEM_WITH_TBB
// #include <tbb/task_scheduler_init.h>
// #endif

#include <igl/write_triangle_mesh.h>
#include <igl/edges.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/connected_components.h>

#include <ipc/ipc.hpp>

#include <tinyxml2.h>

#include <filesystem>

extern "C" size_t getPeakRSS();

namespace polyfem
{
	using namespace assembler;
	using namespace mesh;
	using namespace solver;
	using namespace utils;

	void State::save_timestep(const double time, const int t, const double t0, const double dt)
	{
		if (args["output"]["advanced"]["save_time_sequence"] && !(t % args["output"]["paraview"]["skip_frame"].get<int>()))
		{
			logger().trace("Saving VTU...");
			POLYFEM_SCOPED_TIMER("Saving VTU");
			const std::string step_name = args["output"]["advanced"]["timestep_prefix"];

			if (!solve_export_to_file)
				solution_frames.emplace_back();

			save_vtu(resolve_output_path(fmt::format(step_name + "{:d}.vtu", t)), time);

			save_pvd(
				resolve_output_path(args["output"]["paraview"]["file_name"]),
				[step_name](int i) { return fmt::format(step_name + "{:d}.vtm", i); },
				t, t0, dt, args["output"]["paraview"]["skip_frame"].get<int>());
		}
	}

	void State::get_sidesets(Eigen::MatrixXd &pts, Eigen::MatrixXi &faces, Eigen::MatrixXd &sidesets)
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}

		if (mesh->is_volume())
		{
			const Mesh3D &tmp_mesh = *dynamic_cast<Mesh3D *>(mesh.get());
			int n_pts = 0;
			int n_faces = 0;
			for (int f = 0; f < tmp_mesh.n_faces(); ++f)
			{
				if (tmp_mesh.get_boundary_id(f) > 0)
				{
					n_pts += tmp_mesh.n_face_vertices(f) + 1;
					n_faces += tmp_mesh.n_face_vertices(f);
				}
			}

			pts.resize(n_pts, 3);
			faces.resize(n_faces, 3);
			sidesets.resize(n_pts, 1);

			n_pts = 0;
			n_faces = 0;
			for (int f = 0; f < tmp_mesh.n_faces(); ++f)
			{
				const int sideset = tmp_mesh.get_boundary_id(f);
				if (sideset > 0)
				{
					const int n_face_vertices = tmp_mesh.n_face_vertices(f);

					for (int i = 0; i < n_face_vertices; ++i)
					{
						if (n_face_vertices == 3)
							faces.row(n_faces) << ((i + 1) % n_face_vertices + n_pts), (i + n_pts), (n_pts + n_face_vertices);
						else
							faces.row(n_faces) << (i + n_pts), ((i + 1) % n_face_vertices + n_pts), (n_pts + n_face_vertices);
						++n_faces;
					}

					for (int i = 0; i < n_face_vertices; ++i)
					{
						pts.row(n_pts) = tmp_mesh.point(tmp_mesh.face_vertex(f, i));
						sidesets(n_pts) = sideset;

						++n_pts;
					}

					pts.row(n_pts) = tmp_mesh.face_barycenter(f);
					sidesets(n_pts) = sideset;
					++n_pts;
				}
			}
		}
		else
		{
			const Mesh2D &tmp_mesh = *dynamic_cast<Mesh2D *>(mesh.get());
			int n_siteset = 0;
			for (int e = 0; e < tmp_mesh.n_edges(); ++e)
			{
				if (tmp_mesh.get_boundary_id(e) > 0)
					++n_siteset;
			}

			pts.resize(n_siteset * 2, 2);
			faces.resize(n_siteset, 2);
			sidesets.resize(n_siteset, 1);

			n_siteset = 0;
			for (int e = 0; e < tmp_mesh.n_edges(); ++e)
			{
				const int sideset = tmp_mesh.get_boundary_id(e);
				if (sideset > 0)
				{
					pts.row(2 * n_siteset) = tmp_mesh.point(tmp_mesh.edge_vertex(e, 0));
					pts.row(2 * n_siteset + 1) = tmp_mesh.point(tmp_mesh.edge_vertex(e, 1));
					faces.row(n_siteset) << 2 * n_siteset, 2 * n_siteset + 1;
					sidesets(n_siteset) = sideset;
					++n_siteset;
				}
			}

			pts.conservativeResize(n_siteset * 2, 3);
			pts.col(2).setZero();
		}
	}

	void State::extract_vis_boundary_mesh()
	{
		std::vector<Eigen::MatrixXd> lv, vertices, allnormals;
		std::vector<int> el_ids, global_primitive_ids;
		Eigen::MatrixXd uv, local_pts, tmp_n, normals;
		ElementAssemblyValues vals;
		const auto &sampler = ref_element_sampler;
		const int n_samples = sampler.num_samples();
		int size = 0;

		std::vector<std::pair<int, int>> edges;
		std::vector<std::tuple<int, int, int>> tris;

		const auto &gbases = iso_parametric() ? bases : geom_bases;

		for (auto it = total_local_boundary.begin(); it != total_local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			const auto &gbs = gbases[lb.element_id()];
			const auto &bs = bases[lb.element_id()];

			for (int k = 0; k < lb.size(); ++k)
			{
				switch (lb.type())
				{
				case BoundaryType::TriLine:
					utils::BoundarySampler::normal_for_tri_edge(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_tri_edge(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::QuadLine:
					utils::BoundarySampler::normal_for_quad_edge(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_quad_edge(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::Quad:
					utils::BoundarySampler::normal_for_quad_face(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_quad_face(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::Tri:
					utils::BoundarySampler::normal_for_tri_face(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_tri_face(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::Polygon:
					utils::BoundarySampler::normal_for_polygon_edge(lb[k], lb.global_primitive_id(k), *mesh, tmp_n);
					utils::BoundarySampler::sample_polygon_edge(lb.element_id(), lb.global_primitive_id(k), n_samples, *mesh, uv, local_pts);
					break;
				case BoundaryType::Polyhedron:
					assert(false);
					break;
				case BoundaryType::Invalid:
					assert(false);
					break;
				default:
					assert(false);
				}

				vertices.emplace_back();
				lv.emplace_back(local_pts);
				el_ids.push_back(lb.element_id());
				global_primitive_ids.push_back(lb.global_primitive_id(k));
				gbs.eval_geom_mapping(local_pts, vertices.back());
				vals.compute(lb.element_id(), mesh->is_volume(), local_pts, bs, gbs);
				const int tris_start = tris.size();

				if (mesh->is_volume())
				{
					if (lb.type() == BoundaryType::Quad)
					{
						const auto map = [n_samples, size](int i, int j) { return j * n_samples + i + size; };

						for (int j = 0; j < n_samples - 1; ++j)
						{
							for (int i = 0; i < n_samples - 1; ++i)
							{
								tris.emplace_back(map(i, j), map(i + 1, j), map(i, j + 1));
								tris.emplace_back(map(i + 1, j + 1), map(i, j + 1), map(i + 1, j));
							}
						}
					}
					else if (lb.type() == BoundaryType::Tri)
					{
						int index = 0;
						std::vector<int> mapp(n_samples * n_samples, -1);
						for (int j = 0; j < n_samples; ++j)
						{
							for (int i = 0; i < n_samples - j; ++i)
							{
								mapp[j * n_samples + i] = index;
								++index;
							}
						}
						const auto map = [mapp, n_samples](int i, int j) {
							if (j * n_samples + i >= mapp.size())
								return -1;
							return mapp[j * n_samples + i];
						};

						for (int j = 0; j < n_samples - 1; ++j)
						{
							for (int i = 0; i < n_samples - j; ++i)
							{
								if (map(i, j) >= 0 && map(i + 1, j) >= 0 && map(i, j + 1) >= 0)
									tris.emplace_back(map(i, j) + size, map(i + 1, j) + size, map(i, j + 1) + size);

								if (map(i + 1, j + 1) >= 0 && map(i, j + 1) >= 0 && map(i + 1, j) >= 0)
									tris.emplace_back(map(i + 1, j + 1) + size, map(i, j + 1) + size, map(i + 1, j) + size);
							}
						}
					}
					else
					{
						assert(false);
					}
				}
				else
				{
					for (int i = 0; i < vertices.back().rows() - 1; ++i)
						edges.emplace_back(i + size, i + size + 1);
				}

				normals.resize(vals.jac_it.size(), tmp_n.cols());

				for (int n = 0; n < vals.jac_it.size(); ++n)
				{
					normals.row(n) = tmp_n * vals.jac_it[n];
					normals.row(n).normalize();
				}

				allnormals.push_back(normals);

				tmp_n.setZero();
				for (int n = 0; n < vals.jac_it.size(); ++n)
				{
					tmp_n += normals.row(n);
				}

				if (mesh->is_volume())
				{
					Eigen::Vector3d e1 = vertices.back().row(std::get<1>(tris.back()) - size) - vertices.back().row(std::get<0>(tris.back()) - size);
					Eigen::Vector3d e2 = vertices.back().row(std::get<2>(tris.back()) - size) - vertices.back().row(std::get<0>(tris.back()) - size);

					Eigen::Vector3d n = e1.cross(e2);
					Eigen::Vector3d nn = tmp_n.transpose();

					if (n.dot(nn) < 0)
					{
						for (int i = tris_start; i < tris.size(); ++i)
						{
							tris[i] = std::tuple<int, int, int>(std::get<0>(tris[i]), std::get<2>(tris[i]), std::get<1>(tris[i]));
						}
					}
				}

				size += vertices.back().rows();
			}
		}

		boundary_vis_vertices.resize(size, vertices.front().cols());
		boundary_vis_local_vertices.resize(size, vertices.front().cols());
		boundary_vis_elements_ids.resize(size, 1);
		boundary_vis_primitive_ids.resize(size, 1);
		boundary_vis_normals.resize(size, vertices.front().cols());

		if (mesh->is_volume())
			boundary_vis_elements.resize(tris.size(), 3);
		else
			boundary_vis_elements.resize(edges.size(), 2);

		int index = 0;
		int ii = 0;
		for (const auto &v : vertices)
		{
			boundary_vis_vertices.block(index, 0, v.rows(), v.cols()) = v;
			boundary_vis_local_vertices.block(index, 0, v.rows(), v.cols()) = lv[ii];
			boundary_vis_elements_ids.block(index, 0, v.rows(), 1).setConstant(el_ids[ii]);
			boundary_vis_primitive_ids.block(index, 0, v.rows(), 1).setConstant(global_primitive_ids[ii++]);
			index += v.rows();
		}

		index = 0;
		for (const auto &n : allnormals)
		{
			boundary_vis_normals.block(index, 0, n.rows(), n.cols()) = n;
			index += n.rows();
		}

		index = 0;
		if (mesh->is_volume())
		{
			for (const auto &t : tris)
			{
				boundary_vis_elements.row(index) << std::get<0>(t), std::get<1>(t), std::get<2>(t);
				++index;
			}
		}
		else
		{
			for (const auto &e : edges)
			{
				boundary_vis_elements.row(index) << e.first, e.second;
				++index;
			}
		}
	}

	void State::extract_boundary_mesh(
		const int n_current_bases,
		const std::vector<ElementBases> &bases,
		Eigen::MatrixXd &boundary_nodes_pos,
		Eigen::MatrixXi &boundary_edges,
		Eigen::MatrixXi &boundary_triangles) const
	{
		if (mesh->is_volume())
		{
			boundary_nodes_pos.resize(n_current_bases, 3);
			boundary_nodes_pos.setZero();
			const Mesh3D &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());

			std::vector<std::tuple<int, int, int>> tris;

			for (const LocalBoundary &lb : total_local_boundary)
			{
				const ElementBases &b = bases[lb.element_id()];

				for (int j = 0; j < lb.size(); ++j)
				{
					const int eid = lb.global_primitive_id(j);
					const int lid = lb[j];
					const auto nodes = b.local_nodes_for_primitive(eid, mesh3d);

					if (!mesh->is_simplex(lb.element_id()))
					{
						logger().trace("skipping element {} since it is not a simplex", eid);
						continue;
					}

					std::vector<int> loc_nodes;

					bool is_follower = false;
					if (!mesh3d.is_conforming())
					{
						for (long n = 0; n < nodes.size(); ++n)
						{
							auto &bs = b.bases[nodes(n)];
							const auto &glob = bs.global();
							if (glob.size() != 1)
							{
								is_follower = true;
								break;
							}
						}
					}

					if (is_follower)
						continue;

					for (long n = 0; n < nodes.size(); ++n)
					{
						auto &bs = b.bases[nodes(n)];
						const auto &glob = bs.global();
						if (glob.size() != 1)
							continue;

						int gindex = glob.front().index;
						boundary_nodes_pos.row(gindex) = glob.front().node;
						loc_nodes.push_back(gindex);
					}

					if (loc_nodes.size() == 3)
					{
						tris.emplace_back(loc_nodes[0], loc_nodes[1], loc_nodes[2]);
					}
					else if (loc_nodes.size() == 6)
					{
						tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[5]);
						tris.emplace_back(loc_nodes[3], loc_nodes[1], loc_nodes[4]);
						tris.emplace_back(loc_nodes[4], loc_nodes[2], loc_nodes[5]);
						tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[5]);
					}
					else if (loc_nodes.size() == 10)
					{
						tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[8]);
						tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[9]);
						tris.emplace_back(loc_nodes[4], loc_nodes[1], loc_nodes[5]);
						tris.emplace_back(loc_nodes[5], loc_nodes[6], loc_nodes[9]);
						tris.emplace_back(loc_nodes[6], loc_nodes[2], loc_nodes[7]);
						tris.emplace_back(loc_nodes[7], loc_nodes[8], loc_nodes[9]);
						tris.emplace_back(loc_nodes[8], loc_nodes[3], loc_nodes[9]);
						tris.emplace_back(loc_nodes[9], loc_nodes[4], loc_nodes[5]);
						tris.emplace_back(loc_nodes[6], loc_nodes[7], loc_nodes[9]);
					}
					else if (loc_nodes.size() == 15)
					{
						tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[11]);
						tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[12]);
						tris.emplace_back(loc_nodes[3], loc_nodes[12], loc_nodes[11]);
						tris.emplace_back(loc_nodes[12], loc_nodes[10], loc_nodes[11]);
						tris.emplace_back(loc_nodes[4], loc_nodes[5], loc_nodes[13]);
						tris.emplace_back(loc_nodes[4], loc_nodes[13], loc_nodes[12]);
						tris.emplace_back(loc_nodes[12], loc_nodes[13], loc_nodes[14]);
						tris.emplace_back(loc_nodes[12], loc_nodes[14], loc_nodes[10]);
						tris.emplace_back(loc_nodes[14], loc_nodes[9], loc_nodes[10]);
						tris.emplace_back(loc_nodes[5], loc_nodes[1], loc_nodes[6]);
						tris.emplace_back(loc_nodes[5], loc_nodes[6], loc_nodes[13]);
						tris.emplace_back(loc_nodes[6], loc_nodes[7], loc_nodes[13]);
						tris.emplace_back(loc_nodes[13], loc_nodes[7], loc_nodes[14]);
						tris.emplace_back(loc_nodes[7], loc_nodes[8], loc_nodes[14]);
						tris.emplace_back(loc_nodes[14], loc_nodes[8], loc_nodes[9]);
						tris.emplace_back(loc_nodes[8], loc_nodes[2], loc_nodes[9]);
					}
					else
					{
						logger().warn("Skipping face as it has {} nodes, boundary export supported up to p4", loc_nodes.size());
						// assert(false);
					}
				}
			}

			boundary_triangles.resize(tris.size(), 3);
			for (int i = 0; i < tris.size(); ++i)
			{
				boundary_triangles.row(i) << std::get<0>(tris[i]), std::get<2>(tris[i]), std::get<1>(tris[i]);
			}

			if (boundary_triangles.rows() > 0)
			{
				igl::edges(boundary_triangles, boundary_edges);
			}
		}
		else
		{
			boundary_nodes_pos.resize(n_current_bases, 2);
			boundary_nodes_pos.setZero();
			const Mesh2D &mesh2d = *dynamic_cast<Mesh2D *>(mesh.get());

			std::vector<std::pair<int, int>> edges;

			for (auto it = total_local_boundary.begin(); it != total_local_boundary.end(); ++it)
			{
				const auto &lb = *it;
				const ElementBases &b = bases[lb.element_id()];

				for (int j = 0; j < lb.size(); ++j)
				{
					const int eid = lb.global_primitive_id(j);
					const int lid = lb[j];
					const auto nodes = b.local_nodes_for_primitive(eid, mesh2d);

					int prev_node = -1;

					for (long n = 0; n < nodes.size(); ++n)
					{
						auto &bs = b.bases[nodes(n)];
						const auto &glob = bs.global();
						if (glob.size() != 1)
							continue;

						int gindex = glob.front().index;
						boundary_nodes_pos.row(gindex) << glob.front().node(0), glob.front().node(1);

						if (prev_node >= 0)
							edges.emplace_back(prev_node, gindex);
						prev_node = gindex;
					}
				}
			}

			boundary_triangles.resize(0, 0);
			boundary_edges.resize(edges.size(), 2);
			for (int i = 0; i < edges.size(); ++i)
			{
				boundary_edges.row(i) << edges[i].first, edges[i].second;
			}
		}
	}

	void State::save_json()
	{
		const std::string out_path = resolve_output_path(args["output"]["json"]);
		if (!out_path.empty())
		{
			std::ofstream out(out_path);
			if (!out.is_open())
			{
				logger().error("Unable to save simulation JSON to {}", out_path);
				return;
			}
			save_json(out);
			out.close();
		}
	}

	void State::save_json(std::ostream &out)
	{
		using json = nlohmann::json;
		json j;
		save_json(j);
		out << j.dump(4) << std::endl;
	}

	void State::save_json(nlohmann::json &j)
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (sol.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		logger().info("Saving json...");

		j["args"] = args;

		j["geom_order"] = mesh->orders().size() > 0 ? mesh->orders().maxCoeff() : 1;
		j["geom_order_min"] = mesh->orders().size() > 0 ? mesh->orders().minCoeff() : 1;
		j["discr_order_min"] = disc_orders.minCoeff();
		j["discr_order_max"] = disc_orders.maxCoeff();
		j["iso_parametric"] = iso_parametric();
		j["problem"] = problem->name();
		j["mat_size"] = mat_size;
		j["num_bases"] = n_bases;
		j["num_pressure_bases"] = n_pressure_bases;
		j["num_non_zero"] = nn_zero;
		j["num_flipped"] = n_flipped;
		j["num_dofs"] = num_dofs;
		j["num_vertices"] = mesh->n_vertices();
		j["num_elements"] = mesh->n_elements();

		j["num_p1"] = (disc_orders.array() == 1).count();
		j["num_p2"] = (disc_orders.array() == 2).count();
		j["num_p3"] = (disc_orders.array() == 3).count();
		j["num_p4"] = (disc_orders.array() == 4).count();
		j["num_p5"] = (disc_orders.array() == 5).count();

		j["mesh_size"] = mesh_size;
		j["max_angle"] = max_angle;

		j["sigma_max"] = sigma_max;
		j["sigma_min"] = sigma_min;
		j["sigma_avg"] = sigma_avg;

		j["min_edge_length"] = min_edge_length;
		j["average_edge_length"] = average_edge_length;

		j["err_l2"] = l2_err;
		j["err_h1"] = h1_err;
		j["err_h1_semi"] = h1_semi_err;
		j["err_linf"] = linf_err;
		j["err_linf_grad"] = grad_max_err;
		j["err_lp"] = lp_err;

		j["spectrum"] = {spectrum(0), spectrum(1), spectrum(2), spectrum(3)};
		j["spectrum_condest"] = std::abs(spectrum(3)) / std::abs(spectrum(0));

		// j["errors"] = errors;

		j["time_building_basis"] = building_basis_time;
		j["time_loading_mesh"] = loading_mesh_time;
		j["time_computing_poly_basis"] = computing_poly_basis_time;
		j["time_assembling_stiffness_mat"] = assembling_stiffness_mat_time;
		j["time_assigning_rhs"] = assigning_rhs_time;
		j["time_solving"] = solving_time;
		j["time_computing_errors"] = computing_errors_time;

		j["solver_info"] = solver_info;

		j["count_simplex"] = simplex_count;
		j["count_regular"] = regular_count;
		j["count_regular_boundary"] = regular_boundary_count;
		j["count_simple_singular"] = simple_singular_count;
		j["count_multi_singular"] = multi_singular_count;
		j["count_boundary"] = boundary_count;
		j["count_non_regular_boundary"] = non_regular_boundary_count;
		j["count_non_regular"] = non_regular_count;
		j["count_undefined"] = undefined_count;
		j["count_multi_singular_boundary"] = multi_singular_boundary_count;

		j["is_simplicial"] = mesh->n_elements() == simplex_count;

		j["peak_memory"] = getPeakRSS() / (1024 * 1024);

		const int actual_dim = problem->is_scalar() ? 1 : mesh->dimension();

		std::vector<double> mmin(actual_dim);
		std::vector<double> mmax(actual_dim);

		for (int d = 0; d < actual_dim; ++d)
		{
			mmin[d] = std::numeric_limits<double>::max();
			mmax[d] = -std::numeric_limits<double>::max();
		}

		for (int i = 0; i < sol.size(); i += actual_dim)
		{
			for (int d = 0; d < actual_dim; ++d)
			{
				mmin[d] = std::min(mmin[d], sol(i + d));
				mmax[d] = std::max(mmax[d], sol(i + d));
			}
		}

		std::vector<double> sol_at_node(actual_dim);

		if (args["output"]["advanced"]["sol_at_node"] >= 0)
		{
			const int node_id = args["output"]["advanced"]["sol_at_node"];

			for (int d = 0; d < actual_dim; ++d)
			{
				sol_at_node[d] = sol(node_id * actual_dim + d);
			}
		}

		j["sol_at_node"] = sol_at_node;
		j["sol_min"] = mmin;
		j["sol_max"] = mmax;

#if defined(POLYFEM_WITH_CPP_THREADS)
		j["num_threads"] = polyfem::get_n_threads();
#elif defined(POLYFEM_WITH_TBB)
		j["num_threads"] = get_n_threads();
#else
		j["num_threads"] = 1;
#endif

		j["formulation"] = formulation();

		logger().info("done");
	}

	void State::export_data()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}
		// if (stiffness.rows() <= 0) { logger().error("Assemble the stiffness matrix first!"); return; }
		if (rhs.size() <= 0)
		{
			logger().error("Assemble the rhs first!");
			return;
		}
		if (sol.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		// Export vtu mesh of solution + wire mesh of deformed input
		// + mesh colored with the bases
		const std::string vis_mesh_path = resolve_output_path(args["output"]["paraview"]["file_name"]);
		const std::string nodes_path = resolve_output_path(args["output"]["data"]["nodes"]);
		const std::string solution_path = resolve_output_path(args["output"]["data"]["solution"]);
		const std::string solmat_path = resolve_output_path(args["output"]["data"]["solution_mat"]);
		const std::string stress_path = resolve_output_path(args["output"]["data"]["stress_mat"]);
		const std::string mises_path = resolve_output_path(args["output"]["data"]["mises"]);
		const bool reorder_output = args["output"]["data"]["advanced"]["reorder_nodes"];

		if (!solution_path.empty())
		{
			std::ofstream out(solution_path);
			out.precision(100);
			out << std::scientific;
			if (reorder_output)
			{
				int problem_dim = (problem->is_scalar() ? 1 : mesh->dimension());
				Eigen::VectorXi reordering(n_bases);
				reordering.setConstant(-1);

				for (int i = 0; i < in_node_to_node.size(); ++i)
				{
					reordering[in_node_to_node[i]] = i;
				}
				Eigen::MatrixXd tmp_sol = unflatten(sol, problem_dim);
				Eigen::MatrixXd tmp(tmp_sol.rows(), tmp_sol.cols());

				for (int i = 0; i < reordering.size(); ++i)
				{
					if (reordering[i] < 0)
						continue;

					tmp.row(reordering[i]) = tmp_sol.row(i);
				}

				for (int i = 0; i < tmp.rows(); ++i)
				{
					for (int j = 0; j < tmp.cols(); ++j)
						out << tmp(i, j) << " ";

					out << std::endl;
				}
			}
			else
				out << sol << std::endl;
			out.close();
		}

		double tend = args.value("tend", 1.0); // default=1
		if (tend <= 0)
			tend = 1;

		if (!vis_mesh_path.empty() && args["time"].is_null())
		{
			save_vtu(vis_mesh_path, tend);
		}
		if (!nodes_path.empty())
		{
			MatrixXd nodes(n_bases, mesh->dimension());
			for (const ElementBases &eb : bases)
			{
				for (const Basis &b : eb.bases)
				{
					// for(const auto &lg : b.global())
					for (size_t ii = 0; ii < b.global().size(); ++ii)
					{
						const auto &lg = b.global()[ii];
						nodes.row(lg.index) = lg.node;
					}
				}
			}
			std::ofstream out(nodes_path);
			out.precision(100);
			out << nodes;
			out.close();
		}
		if (!solmat_path.empty())
		{
			Eigen::MatrixXd result;
			int problem_dim = (problem->is_scalar() ? 1 : mesh->dimension());
			compute_vertex_values(problem_dim, bases, sol, result);
			std::ofstream out(solmat_path);
			out.precision(20);
			out << result;
		}
		if (!stress_path.empty())
		{
			Eigen::MatrixXd result;
			Eigen::VectorXd mises;
			compute_stress_at_quadrature_points(sol, result, mises);
			std::ofstream out(stress_path);
			out.precision(20);
			out << result;
		}
		if (!mises_path.empty())
		{
			Eigen::MatrixXd result;
			Eigen::VectorXd mises;
			compute_stress_at_quadrature_points(sol, result, mises);
			std::ofstream out(mises_path);
			out.precision(20);
			out << mises;
		}
	}

	void State::build_vis_mesh(Eigen::MatrixXd &points, Eigen::MatrixXi &tets, Eigen::MatrixXi &el_id, Eigen::MatrixXd &discr)
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}

		const auto &sampler = ref_element_sampler;

		const auto &current_bases = iso_parametric() ? bases : geom_bases;
		int tet_total_size = 0;
		int pts_total_size = 0;

		const bool boundary_only = args["output"]["advanced"]["vis_boundary_only"];

		Eigen::MatrixXd vis_pts_poly;
		Eigen::MatrixXi vis_faces_poly;

		for (size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if (boundary_only && mesh->is_volume() && !mesh->is_boundary_element(i))
				continue;

			if (mesh->is_simplex(i))
			{
				tet_total_size += sampler.simplex_volume().rows();
				pts_total_size += sampler.simplex_points().rows();
			}
			else if (mesh->is_cube(i))
			{
				tet_total_size += sampler.cube_volume().rows();
				pts_total_size += sampler.cube_points().rows();
			}
			else
			{
				if (mesh->is_volume())
				{
					sampler.sample_polyhedron(polys_3d[i].first, polys_3d[i].second, vis_pts_poly, vis_faces_poly);

					tet_total_size += vis_faces_poly.rows();
					pts_total_size += vis_pts_poly.rows();
				}
				else
				{
					sampler.sample_polygon(polys[i], vis_pts_poly, vis_faces_poly);

					tet_total_size += vis_faces_poly.rows();
					pts_total_size += vis_pts_poly.rows();
				}
			}
		}

		points.resize(pts_total_size, mesh->dimension());
		tets.resize(tet_total_size, mesh->is_volume() ? 4 : 3);

		el_id.resize(pts_total_size, 1);
		discr.resize(pts_total_size, 1);

		Eigen::MatrixXd mapped, tmp;
		int tet_index = 0, pts_index = 0;

		for (size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if (boundary_only && mesh->is_volume() && !mesh->is_boundary_element(i))
				continue;

			if (mesh->is_simplex(i))
			{
				bs.eval_geom_mapping(sampler.simplex_points(), mapped);

				tets.block(tet_index, 0, sampler.simplex_volume().rows(), tets.cols()) = sampler.simplex_volume().array() + pts_index;
				tet_index += sampler.simplex_volume().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				discr.block(pts_index, 0, mapped.rows(), 1).setConstant(disc_orders(i));
				el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
				pts_index += mapped.rows();
			}
			else if (mesh->is_cube(i))
			{
				bs.eval_geom_mapping(sampler.cube_points(), mapped);

				tets.block(tet_index, 0, sampler.cube_volume().rows(), tets.cols()) = sampler.cube_volume().array() + pts_index;
				tet_index += sampler.cube_volume().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				discr.block(pts_index, 0, mapped.rows(), 1).setConstant(disc_orders(i));
				el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
				pts_index += mapped.rows();
			}
			else
			{
				if (mesh->is_volume())
				{
					sampler.sample_polyhedron(polys_3d[i].first, polys_3d[i].second, vis_pts_poly, vis_faces_poly);
					bs.eval_geom_mapping(vis_pts_poly, mapped);

					tets.block(tet_index, 0, vis_faces_poly.rows(), tets.cols()) = vis_faces_poly.array() + pts_index;
					tet_index += vis_faces_poly.rows();

					points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
					discr.block(pts_index, 0, mapped.rows(), 1).setConstant(-1);
					el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
					pts_index += mapped.rows();
				}
				else
				{
					sampler.sample_polygon(polys[i], vis_pts_poly, vis_faces_poly);
					bs.eval_geom_mapping(vis_pts_poly, mapped);

					tets.block(tet_index, 0, vis_faces_poly.rows(), tets.cols()) = vis_faces_poly.array() + pts_index;
					tet_index += vis_faces_poly.rows();

					points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
					discr.block(pts_index, 0, mapped.rows(), 1).setConstant(-1);
					el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
					pts_index += mapped.rows();
				}
			}
		}

		assert(pts_index == points.rows());
		assert(tet_index == tets.rows());
	}

	void State::build_high_oder_vis_mesh(Eigen::MatrixXd &points, std::vector<std::vector<int>> &elements, Eigen::MatrixXi &el_id, Eigen::MatrixXd &discr)
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}
		assert(mesh->is_linear());

		std::vector<RowVectorNd> nodes;
		int pts_total_size = 0;
		elements.resize(bases.size());
		Eigen::MatrixXd ref_pts;

		for (size_t i = 0; i < bases.size(); ++i)
		{
			const auto &bs = bases[i];
			if (mesh->is_volume())
			{
				if (mesh->is_simplex(i))
					autogen::p_nodes_3d(disc_orders(i), ref_pts);
				else if (mesh->is_cube(i))
					autogen::q_nodes_3d(disc_orders(i), ref_pts);
				else
					continue;
			}
			else
			{
				if (mesh->is_simplex(i))
					autogen::p_nodes_2d(disc_orders(i), ref_pts);
				else if (mesh->is_cube(i))
					autogen::q_nodes_2d(disc_orders(i), ref_pts);
				else
					continue;
			}

			pts_total_size += ref_pts.rows();
		}

		points.resize(pts_total_size, mesh->dimension());

		el_id.resize(pts_total_size, 1);
		discr.resize(pts_total_size, 1);

		Eigen::MatrixXd mapped;
		int pts_index = 0;

		std::string error_msg = "";

		for (size_t i = 0; i < bases.size(); ++i)
		{
			const auto &bs = bases[i];
			if (mesh->is_volume())
			{
				if (mesh->is_simplex(i))
					autogen::p_nodes_3d(disc_orders(i), ref_pts);
				else if (mesh->is_cube(i))
					autogen::q_nodes_3d(disc_orders(i), ref_pts);
				else
					continue;
			}
			else
			{
				if (mesh->is_simplex(i))
					autogen::p_nodes_2d(disc_orders(i), ref_pts);
				else if (mesh->is_cube(i))
					autogen::q_nodes_2d(disc_orders(i), ref_pts);
				else
					continue;
			}

			bs.eval_geom_mapping(ref_pts, mapped);

			for (int j = 0; j < mapped.rows(); ++j)
			{
				points.row(pts_index) = mapped.row(j);
				el_id(pts_index) = i;
				discr(pts_index) = disc_orders(i);
				elements[i].push_back(pts_index);

				pts_index++;
			}

			if (mesh->is_simplex(i))
			{
				if (mesh->is_volume())
				{
					const int n_nodes = elements[i].size();
					if (disc_orders(i) >= 3)
					{

						std::swap(elements[i][16], elements[i][17]);
						std::swap(elements[i][17], elements[i][18]);
						std::swap(elements[i][18], elements[i][19]);
					}
					if (disc_orders(i) > 4)
						error_msg = "not implementd!!!"; // TODO: higher than 3
				}
				else
				{
					if (disc_orders(i) == 4)
					{
						const int n_nodes = elements[i].size();
						std::swap(elements[i][n_nodes - 1], elements[i][n_nodes - 2]);
					}
					if (disc_orders(i) > 4)
						error_msg = "not implementd!!!"; // TODO: higher than 3
				}
			}
			else
				error_msg = "not implementd!!!"; // TODO: hexes
		}

		if (!error_msg.empty())
			logger().warn(error_msg);

		assert(pts_index == points.rows());
	}

	void State::save_vtu(const std::string &path, const double t)
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}
		// if (stiffness.rows() <= 0) { logger().error("Assemble the stiffness matrix first!"); return; }
		if (rhs.size() <= 0)
		{
			logger().error("Assemble the rhs first!");
			return;
		}
		if (sol.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		const bool export_volume = args["output"]["paraview"]["volume"];
		const bool export_surface = args["output"]["paraview"]["surface"];
		const bool export_wire = args["output"]["paraview"]["wireframe"];
		const bool export_contact_forces = args["output"]["paraview"]["options"]["contact_forces"] && !problem->is_scalar();
		const bool export_friction_forces = args["output"]["paraview"]["options"]["friction_forces"] && !problem->is_scalar();

		const std::filesystem::path fs_path(path);
		const std::string path_stem = fs_path.stem().string();
		const std::string base_path = (fs_path.parent_path() / path_stem).string();

		if (export_volume)
		{
			save_volume(path, t);
		}

		if (export_surface)
		{
			save_surface(base_path + "_surf.vtu");
		}

		if (export_wire)
		{
			save_wire(base_path + "_wire.vtu", t);
		}

		if (!solve_export_to_file)
			return;

		tinyxml2::XMLDocument vtm;
		vtm.InsertEndChild(vtm.NewDeclaration());

		tinyxml2::XMLElement *root = vtm.NewElement("VTKFile");
		vtm.InsertEndChild(root);
		root->SetAttribute("type", "vtkMultiBlockDataSet");
		root->SetAttribute("version", "1.0");

		tinyxml2::XMLElement *multiblock = root->InsertNewChildElement("vtkMultiBlockDataSet");

		if (export_volume)
		{
			tinyxml2::XMLElement *block = multiblock->InsertNewChildElement("Block");
			block->SetAttribute("name", "Volume");
			tinyxml2::XMLElement *dataset = block->InsertNewChildElement("DataSet");
			dataset->SetAttribute("name", "data");
			const std::string tmp(fs_path.filename().string());
			dataset->SetAttribute("file", tmp.c_str());
		}

		if (export_surface)
		{
			tinyxml2::XMLElement *block = multiblock->InsertNewChildElement("Block");
			block->SetAttribute("name", "Surface");

			tinyxml2::XMLElement *dataset = block->InsertNewChildElement("DataSet");
			dataset->SetAttribute("name", "surface");
			dataset->SetAttribute("file", (path_stem + "_surf.vtu").c_str());

			if (export_contact_forces || export_friction_forces)
			{
				tinyxml2::XMLElement *dataset = block->InsertNewChildElement("DataSet");
				dataset->SetAttribute("name", "contact");
				dataset->SetAttribute("file", (path_stem + "_surf_contact.vtu").c_str());
			}
		}

		if (export_wire)
		{
			tinyxml2::XMLElement *block = multiblock->InsertNewChildElement("Block");
			block->SetAttribute("name", "Wireframe");

			tinyxml2::XMLElement *dataset = block->InsertNewChildElement("DataSet");
			dataset->SetAttribute("name", "data");
			dataset->SetAttribute("file", (path_stem + "_wire.vtu").c_str());
		}

		tinyxml2::XMLElement *data_array = root->InsertNewChildElement("FieldData")->InsertNewChildElement("DataArray");
		data_array->SetAttribute("type", "Float32");
		data_array->SetAttribute("Name", "TimeValue");
		data_array->InsertNewText(std::to_string(t).c_str());

		vtm.SaveFile((base_path + ".vtm").c_str());
	}

	void State::save_volume(const std::string &path, const double t)
	{
		Eigen::MatrixXd points;
		Eigen::MatrixXi tets;
		Eigen::MatrixXi el_id;
		Eigen::MatrixXd discr;
		std::vector<std::vector<int>> elements;

		const bool use_sampler = !(mesh->is_linear() && solve_export_to_file && args["output"]["paraview"]["high_order_mesh"]);

		if (use_sampler)
			build_vis_mesh(points, tets, el_id, discr);
		else
			build_high_oder_vis_mesh(points, elements, el_id, discr);

		Eigen::MatrixXd fun, exact_fun, err;
		const bool boundary_only = use_sampler && args["output"]["advanced"]["vis_boundary_only"];
		const bool material_params = args["output"]["paraview"]["options"]["material"];
		const bool body_ids = args["output"]["paraview"]["options"]["body_ids"];
		const bool sol_on_grid = args["output"]["paraview"]["options"]["sol_on_grid"] > 0;
		const bool export_velocity = args["output"]["paraview"]["options"]["velocity"];
		const bool export_acceleration = args["output"]["paraview"]["options"]["acceleration"];

		if (sol_on_grid)
		{
			const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
			Eigen::MatrixXd tmp, tmp_grad;
			Eigen::MatrixXd tmp_p, tmp_grad_p;
			Eigen::MatrixXd res(grid_points_to_elements.size(), problem_dim);
			res.setConstant(std::numeric_limits<double>::quiet_NaN());
			Eigen::MatrixXd res_grad(grid_points_to_elements.size(), problem_dim * problem_dim);
			res_grad.setConstant(std::numeric_limits<double>::quiet_NaN());

			Eigen::MatrixXd res_p(grid_points_to_elements.size(), 1);
			res_p.setConstant(std::numeric_limits<double>::quiet_NaN());
			Eigen::MatrixXd res_grad_p(grid_points_to_elements.size(), problem_dim);
			res_grad_p.setConstant(std::numeric_limits<double>::quiet_NaN());

			for (int i = 0; i < grid_points_to_elements.size(); ++i)
			{
				const int el_id = grid_points_to_elements(i);
				if (el_id < 0)
					continue;
				assert(mesh->is_simplex(el_id));
				const Eigen::MatrixXd bc = grid_points_bc.row(i);
				Eigen::MatrixXd pt(1, bc.cols() - 1);
				for (int d = 1; d < bc.cols(); ++d)
					pt(d - 1) = bc(d);
				interpolate_at_local_vals(el_id, pt, tmp, tmp_grad);

				res.row(i) = tmp;
				res_grad.row(i) = tmp_grad;

				if (assembler.is_mixed(formulation()))
				{
					interpolate_at_local_vals(el_id, 1, pressure_bases, pt, pressure, tmp_p, tmp_grad_p);
					res_p.row(i) = tmp_p;
					res_grad_p.row(i) = tmp_grad_p;
				}
			}

			std::ofstream os(path + "_sol.txt");
			os << res;

			std::ofstream osg(path + "_grad.txt");
			osg << res_grad;

			std::ofstream osgg(path + "_grid.txt");
			osgg << grid_points;

			if (assembler.is_mixed(formulation()))
			{
				std::ofstream osp(path + "_p_sol.txt");
				osp << res_p;

				std::ofstream osgp(path + "_p_grad.txt");
				osgp << res_grad_p;
			}
		}

		interpolate_function(points.rows(), sol, fun, use_sampler, boundary_only);

		if (obstacle.n_vertices() > 0)
		{
			fun.conservativeResize(fun.rows() + obstacle.n_vertices(), fun.cols());
			obstacle.update_displacement(t, fun);
		}

		if (problem->has_exact_sol())
		{
			problem->exact(points, t, exact_fun);
			err = (fun - exact_fun).eval().rowwise().norm();

			if (obstacle.n_vertices() > 0)
			{
				exact_fun.conservativeResize(exact_fun.rows() + obstacle.n_vertices(), exact_fun.cols());
				obstacle.update_displacement(t, exact_fun);

				err.conservativeResize(err.rows() + obstacle.n_vertices(), 1);
				err.bottomRows(obstacle.n_vertices()).setZero();
			}
		}

		VTUWriter writer;

		if (solve_export_to_file && fun.cols() != 1 && !mesh->is_volume())
		{
			fun.conservativeResize(fun.rows(), 3);
			fun.col(2).setZero();

			if (problem->has_exact_sol())
			{
				exact_fun.conservativeResize(exact_fun.rows(), 3);
				exact_fun.col(2).setZero();
			}
		}

		if (solve_export_to_file)
			writer.add_field("solution", fun);
		else
			solution_frames.back().solution = fun;

		if (problem->is_time_dependent())
		{
			bool is_time_integrator_valid = step_data.nl_problem != nullptr && step_data.nl_problem->time_integrator() != nullptr;
			if (export_velocity)
			{
				Eigen::MatrixXd vel = is_time_integrator_valid
										  ? step_data.nl_problem->time_integrator()->v_prev()
										  : Eigen::MatrixXd::Zero(sol.rows(), sol.cols());

				Eigen::MatrixXd interp_vel;
				interpolate_function(points.rows(), vel, interp_vel, use_sampler, boundary_only);
				if (obstacle.n_vertices() > 0)
				{
					interp_vel.conservativeResize(interp_vel.rows() + obstacle.n_vertices(), interp_vel.cols());
					obstacle.set_zero(interp_vel); // TODO
				}

				if (solve_export_to_file)
				{
					writer.add_field("velocity", interp_vel);
				}
				// TODO: else save to solution frames
			}

			if (export_acceleration)
			{
				Eigen::MatrixXd acc = is_time_integrator_valid
										  ? step_data.nl_problem->time_integrator()->a_prev()
										  : Eigen::MatrixXd::Zero(sol.rows(), sol.cols());

				Eigen::MatrixXd interp_acc;
				interpolate_function(points.rows(), acc, interp_acc, use_sampler, boundary_only);
				if (obstacle.n_vertices() > 0)
				{
					interp_acc.conservativeResize(interp_acc.rows() + obstacle.n_vertices(), interp_acc.cols());
					obstacle.set_zero(interp_acc); // TODO
				}

				if (solve_export_to_file)
				{
					writer.add_field("acceleration", interp_acc);
				}
				// TODO: else save to solution frames
			}
		}

		// if(problem->is_mixed())
		if (assembler.is_mixed(formulation()))
		{
			Eigen::MatrixXd interp_p;
			interpolate_function(points.rows(), 1, pressure_bases, pressure, interp_p, use_sampler, boundary_only);

			if (obstacle.n_vertices() > 0)
			{
				interp_p.conservativeResize(interp_p.size() + obstacle.n_vertices(), 1);
				interp_p.bottomRows(obstacle.n_vertices()).setZero();
			}

			if (solve_export_to_file)
				writer.add_field("pressure", interp_p);
			else
				solution_frames.back().pressure = interp_p;
		}

		if (obstacle.n_vertices() > 0)
		{
			discr.conservativeResize(discr.size() + obstacle.n_vertices(), 1);
			discr.bottomRows(obstacle.n_vertices()).setZero();
		}

		if (solve_export_to_file)
			writer.add_field("discr", discr);
		if (problem->has_exact_sol())
		{
			if (solve_export_to_file)
			{
				writer.add_field("exact", exact_fun);
				writer.add_field("error", err);
			}
			else
			{
				solution_frames.back().exact = exact_fun;
				solution_frames.back().error = err;
			}
		}

		if (fun.cols() != 1)
		{
			Eigen::MatrixXd vals, tvals;
			compute_scalar_value(points.rows(), sol, vals, use_sampler, boundary_only);

			if (obstacle.n_vertices() > 0)
			{
				vals.conservativeResize(vals.size() + obstacle.n_vertices(), 1);
				vals.bottomRows(obstacle.n_vertices()).setZero();
			}

			if (solve_export_to_file)
				writer.add_field("scalar_value", vals);
			else
				solution_frames.back().scalar_value = vals;

			if (solve_export_to_file)
			{
				compute_tensor_value(points.rows(), sol, tvals, use_sampler, boundary_only);
				for (int i = 0; i < tvals.cols(); ++i)
				{
					Eigen::MatrixXd tmp = tvals.col(i);
					if (obstacle.n_vertices() > 0)
					{
						tmp.conservativeResize(tmp.size() + obstacle.n_vertices(), 1);
						tmp.bottomRows(obstacle.n_vertices()).setZero();
					}

					const int ii = (i / mesh->dimension()) + 1;
					const int jj = (i % mesh->dimension()) + 1;
					writer.add_field(fmt::format("tensor_value_{:d}{:d}", ii, jj), tmp);
				}
			}

			if (!args["space"]["advanced"]["use_spline"])
			{
				average_grad_based_function(points.rows(), sol, vals, tvals, use_sampler, boundary_only);
				if (obstacle.n_vertices() > 0)
				{
					vals.conservativeResize(vals.size() + obstacle.n_vertices(), 1);
					vals.bottomRows(obstacle.n_vertices()).setZero();
				}

				if (solve_export_to_file)
					writer.add_field("scalar_value_avg", vals);
				else
					solution_frames.back().scalar_value_avg = vals;
				// for(int i = 0; i < tvals.cols(); ++i){
				// 	const int ii = (i / mesh->dimension()) + 1;
				// 	const int jj = (i % mesh->dimension()) + 1;
				// 	writer.add_field("tensor_value_avg_" + std::to_string(ii) + std::to_string(jj), tvals.col(i));
				// }
			}
		}

		if (material_params)
		{
			const LameParameters &params = assembler.lame_params();

			Eigen::MatrixXd lambdas(points.rows(), 1);
			Eigen::MatrixXd mus(points.rows(), 1);
			Eigen::MatrixXd Es(points.rows(), 1);
			Eigen::MatrixXd nus(points.rows(), 1);
			Eigen::MatrixXd rhos(points.rows(), 1);

			Eigen::MatrixXd local_pts;
			Eigen::MatrixXi vis_faces_poly;

			const auto &gbases = iso_parametric() ? bases : geom_bases;

			int index = 0;
			const auto &sampler = ref_element_sampler;
			for (int e = 0; e < int(bases.size()); ++e)
			{
				const ElementBases &gbs = gbases[e];
				const ElementBases &bs = bases[e];

				if (use_sampler)
				{
					if (mesh->is_simplex(e))
						local_pts = sampler.simplex_points();
					else if (mesh->is_cube(e))
						local_pts = sampler.cube_points();
					else
					{
						if (mesh->is_volume())
							sampler.sample_polyhedron(polys_3d[e].first, polys_3d[e].second, local_pts, vis_faces_poly);
						else
							sampler.sample_polygon(polys[e], local_pts, vis_faces_poly);
					}
				}
				else
				{
					if (mesh->is_volume())
					{
						if (mesh->is_simplex(e))
							autogen::p_nodes_3d(disc_orders(e), local_pts);
						else if (mesh->is_cube(e))
							autogen::q_nodes_3d(disc_orders(e), local_pts);
						else
							continue;
					}
					else
					{
						if (mesh->is_simplex(e))
							autogen::p_nodes_2d(disc_orders(e), local_pts);
						else if (mesh->is_cube(e))
							autogen::q_nodes_2d(disc_orders(e), local_pts);
						else
							continue;
					}
				}

				ElementAssemblyValues vals;
				vals.compute(e, mesh->is_volume(), local_pts, bs, gbs);

				for (int j = 0; j < vals.val.rows(); ++j)
				{
					double lambda, mu;

					params.lambda_mu(local_pts.row(j), vals.val.row(j), e, lambda, mu);
					lambdas(index) = lambda;
					mus(index) = mu;

					if (mesh->is_volume())
					{
						Es(index) = mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
						nus(index) = lambda / (2.0 * (lambda + mu));
					}
					else
					{
						Es(index) = 2 * mu * (2.0 * lambda + 2.0 * mu) / (lambda + 2.0 * mu);
						nus(index) = lambda / (lambda + 2.0 * mu);
					}

					rhos(index) = density(local_pts.row(j), vals.val.row(j), e);

					++index;
				}
			}

			assert(index == points.rows());

			if (obstacle.n_vertices() > 0)
			{
				lambdas.conservativeResize(lambdas.size() + obstacle.n_vertices(), 1);
				lambdas.bottomRows(obstacle.n_vertices()).setZero();

				mus.conservativeResize(mus.size() + obstacle.n_vertices(), 1);
				mus.bottomRows(obstacle.n_vertices()).setZero();

				Es.conservativeResize(Es.size() + obstacle.n_vertices(), 1);
				Es.bottomRows(obstacle.n_vertices()).setZero();

				nus.conservativeResize(nus.size() + obstacle.n_vertices(), 1);
				nus.bottomRows(obstacle.n_vertices()).setZero();

				rhos.conservativeResize(rhos.size() + obstacle.n_vertices(), 1);
				rhos.bottomRows(obstacle.n_vertices()).setZero();
			}

			writer.add_field("lambda", lambdas);
			writer.add_field("mu", mus);
			writer.add_field("E", Es);
			writer.add_field("nu", nus);
			writer.add_field("rho", rhos);
		}

		if (body_ids)
		{

			Eigen::MatrixXd ids(points.rows(), 1);

			for (int i = 0; i < points.rows(); ++i)
			{
				ids(i) = mesh->get_body_id(el_id(i));
			}

			if (obstacle.n_vertices() > 0)
			{
				ids.conservativeResize(ids.size() + obstacle.n_vertices(), 1);
				ids.bottomRows(obstacle.n_vertices()).setZero();
			}

			writer.add_field("body_ids", ids);
		}

		// interpolate_function(pts_index, rhs, fun, boundary_only);
		// writer.add_field("rhs", fun);
		if (solve_export_to_file)
		{
			if (obstacle.n_vertices() > 0)
			{
				const int orig_p = points.rows();
				points.conservativeResize(points.rows() + obstacle.n_vertices(), points.cols());
				points.bottomRows(obstacle.n_vertices()) = obstacle.v();

				if (elements.empty())
				{
					for (int i = 0; i < tets.rows(); ++i)
					{
						elements.emplace_back();
						for (int j = 0; j < tets.cols(); ++j)
							elements.back().push_back(tets(i, j));
					}
				}

				for (int i = 0; i < obstacle.get_face_connectivity().rows(); ++i)
				{
					elements.emplace_back();
					for (int j = 0; j < obstacle.get_face_connectivity().cols(); ++j)
						elements.back().push_back(obstacle.get_face_connectivity()(i, j) + orig_p);
				}

				for (int i = 0; i < obstacle.get_edge_connectivity().rows(); ++i)
				{
					elements.emplace_back();
					for (int j = 0; j < obstacle.get_edge_connectivity().cols(); ++j)
						elements.back().push_back(obstacle.get_edge_connectivity()(i, j) + orig_p);
				}

				for (int i = 0; i < obstacle.get_vertex_connectivity().size(); ++i)
				{
					elements.emplace_back();
					elements.back().push_back(obstacle.get_vertex_connectivity()(i) + orig_p);
				}
			}

			if (elements.empty())
				writer.write_mesh(path, points, tets);
			else
				writer.write_mesh(path, points, elements, true);
		}
		else
		{
			solution_frames.back().name = path;
			solution_frames.back().points = points;
			solution_frames.back().connectivity = tets;
		}
	}

	void State::save_surface(const std::string &export_surface)
	{
		const bool material_params = args["output"]["paraview"]["options"]["material"];
		const bool body_ids = args["output"]["paraview"]["options"]["body_ids"];
		const bool export_contact_forces = args["output"]["paraview"]["options"]["contact_forces"] && !problem->is_scalar();
		const bool export_friction_forces = args["output"]["paraview"]["options"]["friction_forces"] && !problem->is_scalar();

		Eigen::MatrixXd fun, interp_p, discr, vect, b_sidesets;

		Eigen::MatrixXd lsol, lp, lgrad, lpgrad;

		int actual_dim = 1;
		if (!problem->is_scalar())
			actual_dim = mesh->dimension();

		discr.resize(boundary_vis_vertices.rows(), 1);
		fun.resize(boundary_vis_vertices.rows(), actual_dim);
		interp_p.resize(boundary_vis_vertices.rows(), 1);
		vect.resize(boundary_vis_vertices.rows(), mesh->dimension());

		b_sidesets.resize(boundary_vis_vertices.rows(), 1);
		b_sidesets.setZero();

		for (int i = 0; i < boundary_vis_vertices.rows(); ++i)
		{
			const auto s_id = mesh->get_boundary_id(boundary_vis_primitive_ids(i));
			if (s_id > 0)
			{
				b_sidesets(i) = s_id;
			}

			const int el_index = boundary_vis_elements_ids(i);
			interpolate_at_local_vals(el_index, boundary_vis_local_vertices.row(i), sol, lsol, lgrad);
			assert(lsol.size() == actual_dim);
			if (assembler.is_mixed(formulation()))
			{
				interpolate_at_local_vals(el_index, 1, pressure_bases, boundary_vis_local_vertices.row(i), pressure, lp, lpgrad);
				assert(lp.size() == 1);
				interp_p(i) = lp(0);
			}

			discr(i) = disc_orders(el_index);
			for (int j = 0; j < actual_dim; ++j)
			{
				fun(i, j) = lsol(j);
			}

			if (actual_dim == 1)
			{
				assert(lgrad.size() == mesh->dimension());
				for (int j = 0; j < mesh->dimension(); ++j)
				{
					vect(i, j) = lgrad(j);
				}
			}
			else
			{
				assert(lgrad.size() == actual_dim * actual_dim);
				Eigen::MatrixXd tensor_flat;
				const auto &gbases = iso_parametric() ? bases : geom_bases;
				const ElementBases &gbs = gbases[el_index];
				const ElementBases &bs = bases[el_index];
				assembler.compute_tensor_value(formulation(), el_index, bs, gbs, boundary_vis_local_vertices.row(i), sol, tensor_flat);
				assert(tensor_flat.size() == actual_dim * actual_dim);
				Map<Eigen::MatrixXd> tensor(tensor_flat.data(), actual_dim, actual_dim);
				vect.row(i) = boundary_vis_normals.row(i) * tensor;
			}
		}

		if (args["contact"]["enabled"] && (export_contact_forces || export_friction_forces) && solve_export_to_file)
		{
			VTUWriter writer;

			const int problem_dim = mesh->dimension();
			Eigen::MatrixXd displaced = unflatten(sol, problem_dim);

			Eigen::MatrixXd real_vertices = collision_mesh.vertices(displaced);
			writer.add_field("solution", real_vertices);

			displaced += boundary_nodes_pos;
			Eigen::MatrixXd displaced_surface = collision_mesh.vertices(displaced);

			ipc::Constraints constraint_set;
			ipc::construct_constraint_set(
				collision_mesh, displaced_surface, args["contact"]["dhat"], constraint_set,
				/*dmin=*/0, ipc::BroadPhaseMethod::HASH_GRID);

			const double barrier_stiffness = step_data.nl_problem != nullptr ? step_data.nl_problem->barrier_stiffness() : 1;

			if (export_contact_forces)
			{
				Eigen::MatrixXd forces = -barrier_stiffness * ipc::compute_barrier_potential_gradient(collision_mesh, displaced_surface, constraint_set, args["contact"]["dhat"]);
				// forces = collision_mesh.to_full_dof(forces);
				// assert(forces.size() == sol.size());

				Eigen::MatrixXd forces_reshaped = unflatten(forces, problem_dim);

				assert(forces_reshaped.rows() == real_vertices.rows());
				assert(forces_reshaped.cols() == real_vertices.cols());
				writer.add_field("contact_forces", forces_reshaped);
			}

			if (export_friction_forces)
			{
				Eigen::MatrixXd displaced_surface_prev =
					(step_data.nl_problem != nullptr)
						? collision_mesh.vertices(step_data.nl_problem->displaced_prev())
						: displaced_surface;

				ipc::FrictionConstraints friction_constraint_set;
				ipc::construct_friction_constraint_set(
					collision_mesh, displaced_surface, constraint_set,
					args["contact"]["dhat"].get<double>(), barrier_stiffness, args["contact"]["friction_coefficient"].get<double>(),
					friction_constraint_set);

				double dt = 1;
				if (!args["time"].is_null())
					dt = args["time"]["dt"];
				Eigen::MatrixXd forces = -ipc::compute_friction_potential_gradient(
					collision_mesh, displaced_surface_prev, displaced_surface,
					friction_constraint_set, args["contact"]["epsv"].get<double>() * dt);
				// forces = collision_mesh.to_full_dof(forces);
				// assert(forces.size() == sol.size());

				Eigen::MatrixXd forces_reshaped = unflatten(forces, problem_dim);

				assert(forces_reshaped.rows() == real_vertices.rows());
				assert(forces_reshaped.cols() == real_vertices.cols());
				writer.add_field("friction_forces", forces_reshaped);
			}

			assert(collision_mesh.vertices(boundary_nodes_pos).rows() == real_vertices.rows());
			assert(collision_mesh.vertices(boundary_nodes_pos).cols() == real_vertices.cols());

			writer.write_mesh(
				export_surface.substr(0, export_surface.length() - 4) + "_contact.vtu",
				collision_mesh.vertices(boundary_nodes_pos),
				problem_dim == 3 ? collision_mesh.faces() : collision_mesh.edges());
		}

		VTUWriter writer;

		if (solve_export_to_file)
		{

			writer.add_field("normals", boundary_vis_normals);
			writer.add_field("solution", fun);
			if (assembler.is_mixed(formulation()))
				writer.add_field("pressure", interp_p);
			writer.add_field("discr", discr);
			writer.add_field("sidesets", b_sidesets);

			if (actual_dim == 1)
				writer.add_field("solution_grad", vect);
			else
				writer.add_field("traction_force", vect);
		}
		else
		{
			solution_frames.back().solution = fun;
			if (assembler.is_mixed(formulation()))
				solution_frames.back().pressure = interp_p;
		}

		if (material_params)
		{
			const LameParameters &params = assembler.lame_params();

			Eigen::MatrixXd lambdas(boundary_vis_vertices.rows(), 1);
			Eigen::MatrixXd mus(boundary_vis_vertices.rows(), 1);
			Eigen::MatrixXd Es(boundary_vis_vertices.rows(), 1);
			Eigen::MatrixXd nus(boundary_vis_vertices.rows(), 1);
			Eigen::MatrixXd rhos(boundary_vis_vertices.rows(), 1);

			for (int i = 0; i < boundary_vis_vertices.rows(); ++i)
			{
				double lambda, mu;

				params.lambda_mu(boundary_vis_local_vertices.row(i), boundary_vis_vertices.row(i), boundary_vis_elements_ids(i), lambda, mu);
				lambdas(i) = lambda;
				mus(i) = mu;
				if (mesh->is_volume())
				{
					Es(i) = mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
					nus(i) = lambda / (2.0 * (lambda + mu));
				}
				else
				{
					Es(i) = 2 * mu * (2.0 * lambda + 2.0 * mu) / (lambda + 2.0 * mu);
					nus(i) = lambda / (lambda + 2.0 * mu);
				}
				rhos(i) = density(boundary_vis_local_vertices.row(i), boundary_vis_vertices.row(i), boundary_vis_elements_ids(i));
			}

			writer.add_field("lambda", lambdas);
			writer.add_field("mu", mus);
			writer.add_field("E", Es);
			writer.add_field("nu", nus);
			writer.add_field("rho", rhos);
		}

		if (body_ids)
		{

			Eigen::MatrixXd ids(boundary_vis_vertices.rows(), 1);

			for (int i = 0; i < boundary_vis_vertices.rows(); ++i)
			{
				ids(i) = mesh->get_body_id(boundary_vis_elements_ids(i));
			}

			writer.add_field("body_ids", ids);
		}
		if (solve_export_to_file)
			writer.write_mesh(export_surface, boundary_vis_vertices, boundary_vis_elements);
		else
		{
			solution_frames.back().name = export_surface;
			solution_frames.back().points = boundary_vis_vertices;
			solution_frames.back().connectivity = boundary_vis_elements;
		}
	}

	void State::save_wire(const std::string &name, const double t)
	{
		if (!solve_export_to_file) // TODO?
			return;
		const auto &sampler = ref_element_sampler;

		const auto &current_bases = iso_parametric() ? bases : geom_bases;
		int seg_total_size = 0;
		int pts_total_size = 0;
		int faces_total_size = 0;

		for (size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if (mesh->is_simplex(i))
			{
				pts_total_size += sampler.simplex_points().rows();
				seg_total_size += sampler.simplex_edges().rows();
				faces_total_size += sampler.simplex_faces().rows();
			}
			else if (mesh->is_cube(i))
			{
				pts_total_size += sampler.cube_points().rows();
				seg_total_size += sampler.cube_edges().rows();
			}
			// TODO add edges for poly
		}

		Eigen::MatrixXd points(pts_total_size, mesh->dimension());
		Eigen::MatrixXi edges(seg_total_size, 2);
		Eigen::MatrixXi faces(faces_total_size, 3);
		points.setZero();

		MatrixXd mapped, tmp;
		int seg_index = 0, pts_index = 0, face_index = 0;
		for (size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if (mesh->is_simplex(i))
			{
				bs.eval_geom_mapping(sampler.simplex_points(), mapped);
				edges.block(seg_index, 0, sampler.simplex_edges().rows(), edges.cols()) = sampler.simplex_edges().array() + pts_index;
				seg_index += sampler.simplex_edges().rows();

				faces.block(face_index, 0, sampler.simplex_faces().rows(), 3) = sampler.simplex_faces().array() + pts_index;
				face_index += sampler.simplex_faces().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				pts_index += mapped.rows();
			}
			else if (mesh->is_cube(i))
			{
				bs.eval_geom_mapping(sampler.cube_points(), mapped);
				edges.block(seg_index, 0, sampler.cube_edges().rows(), edges.cols()) = sampler.cube_edges().array() + pts_index;
				seg_index += sampler.cube_edges().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				pts_index += mapped.rows();
			}
		}

		assert(pts_index == points.rows());
		assert(face_index == faces.rows());

		if (mesh->is_volume())
		{
			// reverse all faces
			for (long i = 0; i < faces.rows(); ++i)
			{
				const int v0 = faces(i, 0);
				const int v1 = faces(i, 1);
				const int v2 = faces(i, 2);

				int tmpc = faces(i, 2);
				faces(i, 2) = faces(i, 1);
				faces(i, 1) = tmpc;
			}
		}
		else
		{
			Matrix2d mmat;
			for (long i = 0; i < faces.rows(); ++i)
			{
				const int v0 = faces(i, 0);
				const int v1 = faces(i, 1);
				const int v2 = faces(i, 2);

				mmat.row(0) = points.row(v2) - points.row(v0);
				mmat.row(1) = points.row(v1) - points.row(v0);

				if (mmat.determinant() > 0)
				{
					int tmpc = faces(i, 2);
					faces(i, 2) = faces(i, 1);
					faces(i, 1) = tmpc;
				}
			}
		}

		Eigen::MatrixXd fun;
		interpolate_function(pts_index, sol, fun, /*use_sampler*/ true, false);

		Eigen::MatrixXd exact_fun, err;

		if (problem->has_exact_sol())
		{
			problem->exact(points, t, exact_fun);
			err = (fun - exact_fun).eval().rowwise().norm();
		}

		if (fun.cols() != 1 && !mesh->is_volume())
		{
			fun.conservativeResize(fun.rows(), 3);
			fun.col(2).setZero();

			exact_fun.conservativeResize(exact_fun.rows(), 3);
			exact_fun.col(2).setZero();
		}

		if (!mesh->is_volume())
		{
			points.conservativeResize(points.rows(), 3);
			points.col(2).setZero();
		}

		VTUWriter writer;
		writer.add_field("solution", fun);
		if (problem->has_exact_sol())
		{
			writer.add_field("exact", exact_fun);
			writer.add_field("error", err);
		}

		if (fun.cols() != 1)
		{
			Eigen::MatrixXd scalar_val;
			compute_scalar_value(pts_index, sol, scalar_val, /*use_sampler*/ true, false);
			writer.add_field("scalar_value", scalar_val);
		}

		writer.write_mesh(name, points, edges);
	}

	void State::save_pvd(const std::string &name, const std::function<std::string(int)> &vtu_names, int time_steps, double t0, double dt, int skip_frame)
	{
		// https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format

		tinyxml2::XMLDocument pvd;
		pvd.InsertEndChild(pvd.NewDeclaration());

		tinyxml2::XMLElement *root = pvd.NewElement("VTKFile");
		pvd.InsertEndChild(root);
		root->SetAttribute("type", "Collection");
		root->SetAttribute("version", "0.1");
		root->SetAttribute("byte_order", "LittleEndian");
		root->SetAttribute("compressor", "vtkZLibDataCompressor");

		tinyxml2::XMLElement *collection = root->InsertNewChildElement("Collection");

		for (int i = 0; i <= time_steps; i += skip_frame)
		{
			tinyxml2::XMLElement *dataset = collection->InsertNewChildElement("DataSet");
			dataset->SetAttribute("timestep", fmt::format("{:g}", t0 + i * dt).c_str());
			dataset->SetAttribute("group", "");
			dataset->SetAttribute("part", "0");
			dataset->SetAttribute("file", vtu_names(i).c_str());
		}

		pvd.SaveFile(name.c_str());
	}

	void State::save_subsolve(const int i, const int t)
	{
		if (!args["output"]["advanced"]["save_solve_sequence_debug"].get<bool>())
			return;

		if (!solve_export_to_file)
			solution_frames.emplace_back();

		save_vtu(resolve_output_path(fmt::format("solve_{:d}.vtu", i)), t);
	}

} // namespace polyfem

#include "OutData.hpp"

#include "Evaluator.hpp"

#include <polyfem/State.hpp>

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/AssemblyValues.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/assembler/Mass.hpp>

#include <polyfem/basis/ElementBases.hpp>

#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>

#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/NLProblem.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/BodyForm.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/ElasticForm.hpp>
#include <polyfem/solver/forms/FrictionForm.hpp>
#include <polyfem/solver/forms/InertiaForm.hpp>
#include <polyfem/solver/forms/LaggedRegForm.hpp>
#include <polyfem/solver/forms/RayleighDampingForm.hpp>
#include <polyfem/solver/forms/BCLagrangianForm.hpp>
#include <polyfem/solver/forms/BCPenaltyForm.hpp>

#include <polyfem/utils/EdgeSampler.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/par_for.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/getRSS.h>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <paraviewo/VTMWriter.hpp>
#include <paraviewo/PVDWriter.hpp>

#include <SimpleBVH/BVH.hpp>

#include <igl/write_triangle_mesh.h>
#include <igl/edges.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/connected_components.h>

#include <ipc/ipc.hpp>

#include <filesystem>

namespace polyfem::io
{
	namespace
	{
		void compute_traction_forces(const State &state, const Eigen::MatrixXd &solution, const double t, Eigen::MatrixXd &traction_forces, bool skip_dirichlet = true)
		{
			int actual_dim = 1;
			if (!state.problem->is_scalar())
				actual_dim = state.mesh->dimension();
			else
				return;

			const std::vector<basis::ElementBases> &bases = state.bases;
			const std::vector<basis::ElementBases> &gbases = state.geom_bases();

			Eigen::MatrixXd uv, samples, gtmp, rhs_fun, deform_mat, trafo;
			Eigen::VectorXi global_primitive_ids;
			Eigen::MatrixXd points, normals;
			Eigen::VectorXd weights;
			polyfem::assembler::ElementAssemblyValues vals;

			traction_forces.setZero(state.n_bases * actual_dim, 1);

			for (const auto &lb : state.total_local_boundary)
			{
				const int e = lb.element_id();
				bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, state.n_boundary_samples(), *state.mesh, false, uv, points, normals, weights, global_primitive_ids);

				if (!has_samples)
					continue;

				const basis::ElementBases &gbs = gbases[e];
				const basis::ElementBases &bs = bases[e];

				vals.compute(e, state.mesh->is_volume(), points, bs, gbs);

				for (int n = 0; n < normals.rows(); ++n)
				{
					trafo = vals.jac_it[n].inverse();

					if (solution.size() > 0)
					{
						assert(actual_dim == 2 || actual_dim == 3);
						deform_mat.resize(actual_dim, actual_dim);
						deform_mat.setZero();
						for (const auto &b : vals.basis_values)
						{
							for (const auto &g : b.global)
							{
								for (int d = 0; d < actual_dim; ++d)
								{
									deform_mat.row(d) += solution(g.index * actual_dim + d) * b.grad.row(n);
								}
							}
						}

						trafo += deform_mat;
					}

					normals.row(n) = normals.row(n) * trafo.inverse();
					normals.row(n).normalize();
				}

				std::vector<assembler::Assembler::NamedMatrix> tensor_flat;
				state.assembler->compute_tensor_value(assembler::OutputData(t, e, bs, gbs, points, solution), tensor_flat);

				for (long n = 0; n < vals.basis_values.size(); ++n)
				{
					const polyfem::assembler::AssemblyValues &v = vals.basis_values[n];

					const int g_index = v.global[0].index * actual_dim;

					for (int q = 0; q < points.rows(); ++q)
					{
						// TF computed only from cauchy stress
						assert(tensor_flat[0].first == "cauchy_stess");
						assert(tensor_flat[0].second.row(q).size() == actual_dim * actual_dim);

						Eigen::MatrixXd stress_tensor = utils::unflatten(tensor_flat[0].second.row(q), actual_dim);

						traction_forces.block(g_index, 0, actual_dim, 1) += stress_tensor * normals.row(q).transpose() * v.val(q) * weights(q);
					}
				}
			}
		}
	} // namespace

	void OutGeometryData::extract_boundary_mesh(
		const mesh::Mesh &mesh,
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		Eigen::MatrixXd &node_positions,
		Eigen::MatrixXi &boundary_edges,
		Eigen::MatrixXi &boundary_triangles,
		std::vector<Eigen::Triplet<double>> &displacement_map_entries)
	{
		using namespace polyfem::mesh;

		displacement_map_entries.clear();

		if (mesh.is_volume())
		{
			if (mesh.has_poly())
			{
				logger().warn("Skipping as the mesh has polygons");
				return;
			}

			const bool is_simplicial = mesh.is_simplicial();

			node_positions.resize(n_bases + (is_simplicial ? 0 : mesh.n_faces()), 3);
			node_positions.setZero();
			const Mesh3D &mesh3d = dynamic_cast<const Mesh3D &>(mesh);

			std::vector<std::tuple<int, int, int>> tris;

			std::vector<bool> visited_node(n_bases, false);

			std::stringstream print_warning;

			for (const LocalBoundary &lb : total_local_boundary)
			{
				const basis::ElementBases &b = bases[lb.element_id()];

				for (int j = 0; j < lb.size(); ++j)
				{
					const int eid = lb.global_primitive_id(j);
					const int lid = lb[j];
					const Eigen::VectorXi nodes = b.local_nodes_for_primitive(eid, mesh3d);

					if (mesh.is_cube(lb.element_id()))
					{
						assert(!is_simplicial);
						assert(!mesh.has_poly());
						std::vector<int> loc_nodes;
						RowVectorNd bary = RowVectorNd::Zero(3);

						for (long n = 0; n < nodes.size(); ++n)
						{
							auto &bs = b.bases[nodes(n)];
							const auto &glob = bs.global();
							if (glob.size() != 1)
								continue;

							int gindex = glob.front().index;
							node_positions.row(gindex) = glob.front().node;
							bary += glob.front().node;
							loc_nodes.push_back(gindex);
						}

						if (loc_nodes.size() != 4)
						{
							logger().trace("skipping element {} since it is not Q1", eid);
							continue;
						}

						bary /= 4;

						const int new_node = n_bases + eid;
						node_positions.row(new_node) = bary;
						tris.emplace_back(loc_nodes[1], loc_nodes[0], new_node);
						tris.emplace_back(loc_nodes[2], loc_nodes[1], new_node);
						tris.emplace_back(loc_nodes[3], loc_nodes[2], new_node);
						tris.emplace_back(loc_nodes[0], loc_nodes[3], new_node);

						for (int q = 0; q < 4; ++q)
						{
							if (!visited_node[loc_nodes[q]])
								displacement_map_entries.emplace_back(loc_nodes[q], loc_nodes[q], 1);

							visited_node[loc_nodes[q]] = true;
							displacement_map_entries.emplace_back(new_node, loc_nodes[q], 0.25);
						}

						continue;
					}

					if (!mesh.is_simplex(lb.element_id()))
					{
						logger().trace("skipping element {} since it is not a simplex or hex", eid);
						continue;
					}

					assert(mesh.is_simplex(lb.element_id()));

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
						const basis::Basis &bs = b.bases[nodes(n)];
						const std::vector<basis::Local2Global> &glob = bs.global();
						if (glob.size() != 1)
							continue;

						int gindex = glob.front().index;
						node_positions.row(gindex) = glob.front().node;
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
						print_warning << loc_nodes.size() << " ";
						// assert(false);
					}

					if (!is_simplicial)
					{
						for (int k = 0; k < loc_nodes.size(); ++k)
						{
							if (!visited_node[loc_nodes[k]])
								displacement_map_entries.emplace_back(loc_nodes[k], loc_nodes[k], 1);

							visited_node[loc_nodes[k]] = true;
						}
					}
				}
			}

			if (print_warning.str().size() > 0)
				logger().warn("Skipping faces as theys have {} nodes, boundary export supported up to p4", print_warning.str());

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
			node_positions.resize(n_bases, 2);
			node_positions.setZero();
			const Mesh2D &mesh2d = dynamic_cast<const Mesh2D &>(mesh);

			std::vector<std::pair<int, int>> edges;

			for (const LocalBoundary &lb : total_local_boundary)
			{
				const basis::ElementBases &b = bases[lb.element_id()];

				for (int j = 0; j < lb.size(); ++j)
				{
					const int eid = lb.global_primitive_id(j);
					const int lid = lb[j];
					const Eigen::VectorXi nodes = b.local_nodes_for_primitive(eid, mesh2d);

					int prev_node = -1;

					for (long n = 0; n < nodes.size(); ++n)
					{
						const basis::Basis &bs = b.bases[nodes(n)];
						const std::vector<basis::Local2Global> &glob = bs.global();
						if (glob.size() != 1)
							continue;

						int gindex = glob.front().index;
						node_positions.row(gindex) = glob.front().node.head<2>();

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

	void OutGeometryData::build_vis_boundary_mesh(
		const mesh::Mesh &mesh,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		const Eigen::MatrixXd &solution,
		const int problem_dim,
		Eigen::MatrixXd &boundary_vis_vertices,
		Eigen::MatrixXd &boundary_vis_local_vertices,
		Eigen::MatrixXi &boundary_vis_elements,
		Eigen::MatrixXi &boundary_vis_elements_ids,
		Eigen::MatrixXi &boundary_vis_primitive_ids,
		Eigen::MatrixXd &boundary_vis_normals,
		Eigen::MatrixXd &displaced_boundary_vis_normals) const
	{
		using namespace polyfem::mesh;

		std::vector<Eigen::MatrixXd> lv, vertices, allnormals, displaced_allnormals;
		std::vector<int> el_ids, global_primitive_ids;
		Eigen::MatrixXd uv, local_pts, tmp_n, normals, displaced_normals, trafo, deform_mat;
		assembler::ElementAssemblyValues vals;
		const auto &sampler = ref_element_sampler;
		const int n_samples = sampler.num_samples();
		int size = 0;

		std::vector<std::pair<int, int>> edges;
		std::vector<std::tuple<int, int, int>> tris;

		for (auto it = total_local_boundary.begin(); it != total_local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			const auto &gbs = gbases[lb.element_id()];
			const auto &bs = bases[lb.element_id()];

			for (int k = 0; k < lb.size(); ++k)
			{
				switch (lb.type())
				{
				case BoundaryType::TRI_LINE:
					utils::BoundarySampler::normal_for_tri_edge(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_tri_edge(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::QUAD_LINE:
					utils::BoundarySampler::normal_for_quad_edge(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_quad_edge(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::QUAD:
					utils::BoundarySampler::normal_for_quad_face(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_quad_face(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::TRI:
					utils::BoundarySampler::normal_for_tri_face(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_tri_face(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::POLYGON:
					utils::BoundarySampler::normal_for_polygon_edge(lb.element_id(), lb.global_primitive_id(k), mesh, tmp_n);
					utils::BoundarySampler::sample_polygon_edge(lb.element_id(), lb.global_primitive_id(k), n_samples, mesh, uv, local_pts);
					break;
				case BoundaryType::POLYHEDRON:
					assert(false);
					break;
				case BoundaryType::INVALID:
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
				vals.compute(lb.element_id(), mesh.is_volume(), local_pts, bs, gbs);
				const int tris_start = tris.size();

				if (mesh.is_volume())
				{
					if (lb.type() == BoundaryType::QUAD)
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
					else if (lb.type() == BoundaryType::TRI)
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
				displaced_normals.resize(vals.jac_it.size(), tmp_n.cols());

				for (int n = 0; n < vals.jac_it.size(); ++n)
				{
					trafo = vals.jac_it[n].inverse();

					if (problem_dim == 2 || problem_dim == 3)
					{

						if (solution.size() > 0)
						{
							deform_mat.resize(problem_dim, problem_dim);
							deform_mat.setZero();
							for (const auto &b : vals.basis_values)
								for (const auto &g : b.global)
									for (int d = 0; d < problem_dim; ++d)
										deform_mat.row(d) += solution(g.index * problem_dim + d) * b.grad.row(n);

							trafo += deform_mat;
						}
					}

					normals.row(n) = tmp_n * vals.jac_it[n];
					normals.row(n).normalize();

					displaced_normals.row(n) = tmp_n * trafo.inverse();
					displaced_normals.row(n).normalize();
				}

				allnormals.push_back(normals);
				displaced_allnormals.push_back(displaced_normals);

				tmp_n.setZero();
				for (int n = 0; n < vals.jac_it.size(); ++n)
				{
					tmp_n += normals.row(n);
				}

				if (mesh.is_volume())
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
		displaced_boundary_vis_normals.resize(size, vertices.front().cols());

		if (mesh.is_volume())
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
		for (const auto &n : displaced_allnormals)
		{
			displaced_boundary_vis_normals.block(index, 0, n.rows(), n.cols()) = n;
			index += n.rows();
		}

		index = 0;
		if (mesh.is_volume())
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

	void OutGeometryData::build_vis_mesh(
		const mesh::Mesh &mesh,
		const Eigen::VectorXi &disc_orders,
		const std::vector<basis::ElementBases> &gbases,
		const std::map<int, Eigen::MatrixXd> &polys,
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
		const bool boundary_only,
		Eigen::MatrixXd &points,
		Eigen::MatrixXi &tets,
		Eigen::MatrixXi &el_id,
		Eigen::MatrixXd &discr) const
	{
		// if (!mesh)
		// {
		// 	logger().error("Load the mesh first!");
		// 	return;
		// }
		// if (n_bases <= 0)
		// {
		// 	logger().error("Build the bases first!");
		// 	return;
		// }

		const auto &sampler = ref_element_sampler;

		const auto &current_bases = gbases;
		int tet_total_size = 0;
		int pts_total_size = 0;

		Eigen::MatrixXd vis_pts_poly;
		Eigen::MatrixXi vis_faces_poly, vis_edges_poly;

		for (size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if (boundary_only && mesh.is_volume() && !mesh.is_boundary_element(i))
				continue;

			if (mesh.is_simplex(i))
			{
				tet_total_size += sampler.simplex_volume().rows();
				pts_total_size += sampler.simplex_points().rows();
			}
			else if (mesh.is_cube(i))
			{
				tet_total_size += sampler.cube_volume().rows();
				pts_total_size += sampler.cube_points().rows();
			}
			else
			{
				if (mesh.is_volume())
				{
					sampler.sample_polyhedron(polys_3d.at(i).first, polys_3d.at(i).second, vis_pts_poly, vis_faces_poly, vis_edges_poly);

					tet_total_size += vis_faces_poly.rows();
					pts_total_size += vis_pts_poly.rows();
				}
				else
				{
					sampler.sample_polygon(polys.at(i), vis_pts_poly, vis_faces_poly, vis_edges_poly);

					tet_total_size += vis_faces_poly.rows();
					pts_total_size += vis_pts_poly.rows();
				}
			}
		}

		points.resize(pts_total_size, mesh.dimension());
		tets.resize(tet_total_size, mesh.is_volume() ? 4 : 3);

		el_id.resize(pts_total_size, 1);
		discr.resize(pts_total_size, 1);

		Eigen::MatrixXd mapped, tmp;
		int tet_index = 0, pts_index = 0;

		for (size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if (boundary_only && mesh.is_volume() && !mesh.is_boundary_element(i))
				continue;

			if (mesh.is_simplex(i))
			{
				bs.eval_geom_mapping(sampler.simplex_points(), mapped);

				tets.block(tet_index, 0, sampler.simplex_volume().rows(), tets.cols()) = sampler.simplex_volume().array() + pts_index;
				tet_index += sampler.simplex_volume().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				discr.block(pts_index, 0, mapped.rows(), 1).setConstant(disc_orders(i));
				el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
				pts_index += mapped.rows();
			}
			else if (mesh.is_cube(i))
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
				if (mesh.is_volume())
				{
					sampler.sample_polyhedron(polys_3d.at(i).first, polys_3d.at(i).second, vis_pts_poly, vis_faces_poly, vis_edges_poly);
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
					sampler.sample_polygon(polys.at(i), vis_pts_poly, vis_faces_poly, vis_edges_poly);
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

	void OutGeometryData::build_high_order_vis_mesh(
		const mesh::Mesh &mesh,
		const Eigen::VectorXi &disc_orders,
		const std::vector<basis::ElementBases> &bases,
		Eigen::MatrixXd &points,
		std::vector<std::vector<int>> &elements,
		Eigen::MatrixXi &el_id,
		Eigen::MatrixXd &discr) const
	{
		// if (!mesh)
		// {
		// 	logger().error("Load the mesh first!");
		// 	return;
		// }
		// if (n_bases <= 0)
		// {
		// 	logger().error("Build the bases first!");
		// 	return;
		// }
		assert(mesh.is_linear());

		std::vector<RowVectorNd> nodes;
		int pts_total_size = 0;
		elements.resize(bases.size());
		Eigen::MatrixXd ref_pts;

		for (size_t i = 0; i < bases.size(); ++i)
		{
			const auto &bs = bases[i];
			if (mesh.is_volume())
			{
				if (mesh.is_simplex(i))
					autogen::p_nodes_3d(disc_orders(i), ref_pts);
				else if (mesh.is_cube(i))
					autogen::q_nodes_3d(disc_orders(i), ref_pts);
				else
					continue;
			}
			else
			{
				if (mesh.is_simplex(i))
					autogen::p_nodes_2d(disc_orders(i), ref_pts);
				else if (mesh.is_cube(i))
					autogen::q_nodes_2d(disc_orders(i), ref_pts);
				else
				{
					const int n_v = static_cast<const mesh::Mesh2D &>(mesh).n_face_vertices(i);
					ref_pts.resize(n_v, 2);
				}
			}

			pts_total_size += ref_pts.rows();
		}

		points.resize(pts_total_size, mesh.dimension());

		el_id.resize(pts_total_size, 1);
		discr.resize(pts_total_size, 1);

		Eigen::MatrixXd mapped;
		int pts_index = 0;

		std::string error_msg = "";

		for (size_t i = 0; i < bases.size(); ++i)
		{
			const auto &bs = bases[i];
			if (mesh.is_volume())
			{
				if (mesh.is_simplex(i))
					autogen::p_nodes_3d(disc_orders(i), ref_pts);
				else if (mesh.is_cube(i))
					autogen::q_nodes_3d(disc_orders(i), ref_pts);
				else
					continue;
			}
			else
			{
				if (mesh.is_simplex(i))
					autogen::p_nodes_2d(disc_orders(i), ref_pts);
				else if (mesh.is_cube(i))
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

			if (mesh.is_simplex(i))
			{
				if (mesh.is_volume())
				{
					const int n_nodes = elements[i].size();
					if (disc_orders(i) >= 3)
					{
						std::swap(elements[i][16], elements[i][17]);
						std::swap(elements[i][17], elements[i][18]);
						std::swap(elements[i][18], elements[i][19]);
					}
					if (disc_orders(i) > 4)
						error_msg = "Saving high-order meshes not implemented for P5+ elements!";
				}
				else
				{
					if (disc_orders(i) == 4)
					{
						const int n_nodes = elements[i].size();
						std::swap(elements[i][n_nodes - 1], elements[i][n_nodes - 2]);
					}
					if (disc_orders(i) > 4)
						error_msg = "Saving high-order meshes not implemented for P5+ elements!";
				}
			}
			else if (disc_orders(i) > 1)
				error_msg = "Saving high-order meshes not implemented for Q2+ elements!";
		}

		if (!error_msg.empty())
			logger().warn(error_msg);

		for (size_t i = 0; i < bases.size(); ++i)
		{
			if (mesh.is_volume() || !mesh.is_polytope(i))
				continue;

			const auto &mesh2d = static_cast<const mesh::Mesh2D &>(mesh);
			const int n_v = mesh2d.n_face_vertices(i);

			for (int j = 0; j < n_v; ++j)
			{
				points.row(pts_index) = mesh2d.point(mesh2d.face_vertex(i, j));
				el_id(pts_index) = i;
				discr(pts_index) = disc_orders(i);
				elements[i].push_back(pts_index);

				pts_index++;
			}
		}

		assert(pts_index == points.rows());
	}

	void OutGeometryData::export_data(
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &pressure,
		const bool is_time_dependent,
		const double tend_in,
		const double dt,
		const ExportOptions &opts,
		const std::string &vis_mesh_path,
		const std::string &nodes_path,
		const std::string &solution_path,
		const std::string &stress_path,
		const std::string &mises_path,
		const bool is_contact_enabled,
		std::vector<SolutionFrame> &solution_frames) const
	{
		if (!state.mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		const int n_bases = state.n_bases;
		const std::vector<basis::ElementBases> &bases = state.bases;
		const std::vector<basis::ElementBases> &gbases = state.geom_bases();
		const mesh::Mesh &mesh = *state.mesh;
		const Eigen::VectorXi &in_node_to_node = state.in_node_to_node;
		const Eigen::MatrixXd &rhs = state.rhs;
		const assembler::Problem &problem = *state.problem;

		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}
		// if (rhs.size() <= 0)
		// {
		// 	logger().error("Assemble the rhs first!");
		// 	return;
		// }
		if (sol.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		if (!solution_path.empty())
		{
			std::ofstream out(solution_path);
			out.precision(100);
			out << std::scientific;
			if (opts.reorder_output)
			{
				int problem_dim = (problem.is_scalar() ? 1 : mesh.dimension());
				Eigen::VectorXi reordering(n_bases);
				reordering.setConstant(-1);

				for (int i = 0; i < in_node_to_node.size(); ++i)
				{
					reordering[in_node_to_node[i]] = i;
				}
				Eigen::MatrixXd tmp_sol = utils::unflatten(sol, problem_dim);
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

		double tend = tend_in;
		if (tend <= 0)
			tend = 1;

		if (!vis_mesh_path.empty() && !is_time_dependent)
		{
			save_vtu(
				vis_mesh_path, state, sol, pressure,
				tend, dt, opts,
				is_contact_enabled, solution_frames);
		}
		if (!nodes_path.empty())
		{
			Eigen::MatrixXd nodes(n_bases, mesh.dimension());
			for (const basis::ElementBases &eb : bases)
			{
				for (const basis::Basis &b : eb.bases)
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
		if (!stress_path.empty())
		{
			Eigen::MatrixXd result;
			Eigen::VectorXd mises;
			Evaluator::compute_stress_at_quadrature_points(
				mesh, problem.is_scalar(),
				bases, gbases, state.disc_orders, *state.assembler,
				sol, tend, result, mises);
			std::ofstream out(stress_path);
			out.precision(20);
			out << result;
		}
		if (!mises_path.empty())
		{
			Eigen::MatrixXd result;
			Eigen::VectorXd mises;
			Evaluator::compute_stress_at_quadrature_points(
				mesh, problem.is_scalar(),
				bases, gbases, state.disc_orders, *state.assembler,
				sol, tend, result, mises);
			std::ofstream out(mises_path);
			out.precision(20);
			out << mises;
		}
	}

	OutGeometryData::ExportOptions::ExportOptions(const json &args, const bool is_mesh_linear, const bool is_problem_scalar, const bool solve_export_to_file)
	{
		volume = args["output"]["paraview"]["volume"];
		surface = args["output"]["paraview"]["surface"];
		wire = args["output"]["paraview"]["wireframe"];
		points = args["output"]["paraview"]["points"];
		contact_forces = args["output"]["paraview"]["options"]["contact_forces"] && !is_problem_scalar;
		friction_forces = args["output"]["paraview"]["options"]["friction_forces"] && !is_problem_scalar;

		use_sampler = !(is_mesh_linear && solve_export_to_file && args["output"]["paraview"]["high_order_mesh"]);
		boundary_only = use_sampler && args["output"]["advanced"]["vis_boundary_only"];
		material_params = args["output"]["paraview"]["options"]["material"];
		body_ids = args["output"]["paraview"]["options"]["body_ids"];
		sol_on_grid = args["output"]["advanced"]["sol_on_grid"] > 0;
		velocity = args["output"]["paraview"]["options"]["velocity"];
		acceleration = args["output"]["paraview"]["options"]["acceleration"];
		forces = args["output"]["paraview"]["options"]["forces"] && !is_problem_scalar;

		scalar_values = args["output"]["paraview"]["options"]["scalar_values"];
		tensor_values = args["output"]["paraview"]["options"]["tensor_values"] && !is_problem_scalar;
		discretization_order = args["output"]["paraview"]["options"]["discretization_order"];
		nodes = args["output"]["paraview"]["options"]["nodes"] && !is_problem_scalar;

		use_spline = args["space"]["basis_type"] == "Spline";

		reorder_output = args["output"]["data"]["advanced"]["reorder_nodes"];

		use_hdf5 = args["output"]["paraview"]["options"]["use_hdf5"];

		this->solve_export_to_file = solve_export_to_file;
	}

	void OutGeometryData::save_vtu(
		const std::string &path,
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &pressure,
		const double t,
		const double dt,
		const ExportOptions &opts,
		const bool is_contact_enabled,
		std::vector<SolutionFrame> &solution_frames) const
	{
		if (!state.mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		const mesh::Mesh &mesh = *state.mesh;
		const Eigen::MatrixXd &rhs = state.rhs;

		if (state.n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}
		// if (rhs.size() <= 0)
		// {
		// 	logger().error("Assemble the rhs first!");
		// 	return;
		// }
		if (sol.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		const std::filesystem::path fs_path(path);
		const std::string path_stem = fs_path.stem().string();
		const std::string base_path = (fs_path.parent_path() / path_stem).string();

		if (opts.volume)
		{
			save_volume(base_path + opts.file_extension(), state, sol, pressure, t, dt, opts, solution_frames);
		}

		if (opts.surface)
		{
			save_surface(base_path + "_surf" + opts.file_extension(), state, sol, pressure, t, dt, opts,
						 is_contact_enabled, solution_frames);
		}

		if (is_contact_enabled && (opts.contact_forces || opts.friction_forces))
		{
			save_contact_surface(base_path + "_surf" + opts.file_extension(), state, sol, pressure, t, dt, opts,
								 is_contact_enabled, solution_frames);
		}

		if (opts.wire)
		{
			save_wire(base_path + "_wire" + opts.file_extension(), state, sol, t, opts, solution_frames);
		}

		if (opts.points)
		{
			save_points(base_path + "_points" + opts.file_extension(), state, sol, opts, solution_frames);
		}

		if (!opts.solve_export_to_file)
			return;

		paraviewo::VTMWriter vtm(t);
		if (opts.volume)
			vtm.add_dataset("Volume", "data", path_stem + opts.file_extension());
		if (opts.surface)
			vtm.add_dataset("Surface", "data", path_stem + "_surf" + opts.file_extension());
		if (is_contact_enabled && (opts.contact_forces || opts.friction_forces))
			vtm.add_dataset("Contact", "data", path_stem + "_surf_contact" + opts.file_extension());
		if (opts.wire)
			vtm.add_dataset("Wireframe", "data", path_stem + "_wire" + opts.file_extension());
		if (opts.points)
			vtm.add_dataset("Points", "data", path_stem + "_points" + opts.file_extension());
		vtm.save(base_path + ".vtm");
	}

	void OutGeometryData::save_volume(
		const std::string &path,
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &pressure,
		const double t,
		const double dt,
		const ExportOptions &opts,
		std::vector<SolutionFrame> &solution_frames) const
	{
		const Eigen::VectorXi &disc_orders = state.disc_orders;
		const auto &density = state.mass_matrix_assembler->density();
		const std::vector<basis::ElementBases> &bases = state.bases;
		const std::vector<basis::ElementBases> &pressure_bases = state.pressure_bases;
		const std::vector<basis::ElementBases> &gbases = state.geom_bases();
		const std::map<int, Eigen::MatrixXd> &polys = state.polys;
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d = state.polys_3d;
		const assembler::Assembler &assembler = *state.assembler;
		const std::shared_ptr<time_integrator::ImplicitTimeIntegrator> time_integrator = state.solve_data.time_integrator;
		const mesh::Mesh &mesh = *state.mesh;
		const mesh::Obstacle &obstacle = state.obstacle;
		const assembler::Problem &problem = *state.problem;

		Eigen::MatrixXd points;
		Eigen::MatrixXi tets;
		Eigen::MatrixXi el_id;
		Eigen::MatrixXd discr;
		std::vector<std::vector<int>> elements;

		if (opts.use_sampler)
			build_vis_mesh(mesh, disc_orders, gbases,
						   state.polys, state.polys_3d, opts.boundary_only,
						   points, tets, el_id, discr);
		else
			build_high_order_vis_mesh(mesh, disc_orders, bases,
									  points, elements, el_id, discr);

		Eigen::MatrixXd fun, exact_fun, err, node_fun;

		if (opts.sol_on_grid)
		{
			const int problem_dim = problem.is_scalar() ? 1 : mesh.dimension();
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
				assert(mesh.is_simplex(el_id));
				const Eigen::MatrixXd bc = grid_points_bc.row(i);
				Eigen::MatrixXd pt(1, bc.cols() - 1);
				for (int d = 1; d < bc.cols(); ++d)
					pt(d - 1) = bc(d);
				Evaluator::interpolate_at_local_vals(
					mesh, problem.is_scalar(), bases, gbases,
					el_id, pt, sol, tmp, tmp_grad);

				res.row(i) = tmp;
				res_grad.row(i) = tmp_grad;

				if (state.mixed_assembler != nullptr)
				{
					Evaluator::interpolate_at_local_vals(
						mesh, 1, pressure_bases, gbases,
						el_id, pt, pressure, tmp_p, tmp_grad_p);
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

			if (state.mixed_assembler != nullptr)
			{
				std::ofstream osp(path + "_p_sol.txt");
				osp << res_p;

				std::ofstream osgp(path + "_p_grad.txt");
				osgp << res_grad_p;
			}
		}

		Evaluator::interpolate_function(
			mesh, problem.is_scalar(), bases, state.disc_orders,
			state.polys, state.polys_3d, ref_element_sampler,
			points.rows(), sol, fun, opts.use_sampler, opts.boundary_only);

		{
			Eigen::MatrixXd tmp = Eigen::VectorXd::LinSpaced(sol.size(), 0, sol.size() - 1);

			Evaluator::interpolate_function(
				mesh, problem.is_scalar(), bases, state.disc_orders,
				state.polys, state.polys_3d, ref_element_sampler,
				points.rows(), tmp, node_fun, opts.use_sampler, opts.boundary_only);
		}

		if (obstacle.n_vertices() > 0)
		{
			fun.conservativeResize(fun.rows() + obstacle.n_vertices(), fun.cols());
			node_fun.conservativeResize(node_fun.rows() + obstacle.n_vertices(), node_fun.cols());
			node_fun.bottomRows(obstacle.n_vertices()).setZero();
			// obstacle.update_displacement(t, fun);
			// NOTE: Assuming the obstacle displacement is the last part of the solution
			fun.bottomRows(obstacle.n_vertices()) = utils::unflatten(sol.bottomRows(obstacle.ndof()), fun.cols());
		}

		if (problem.has_exact_sol())
		{
			problem.exact(points, t, exact_fun);
			err = (fun - exact_fun).eval().rowwise().norm();

			if (obstacle.n_vertices() > 0)
			{
				exact_fun.conservativeResize(exact_fun.rows() + obstacle.n_vertices(), exact_fun.cols());
				// obstacle.update_displacement(t, exact_fun);
				exact_fun.bottomRows(obstacle.n_vertices()) = utils::unflatten(sol.bottomRows(obstacle.ndof()), fun.cols());

				err.conservativeResize(err.rows() + obstacle.n_vertices(), 1);
				err.bottomRows(obstacle.n_vertices()).setZero();
			}
		}

		std::shared_ptr<paraviewo::ParaviewWriter> tmpw;
		if (opts.use_hdf5)
			tmpw = std::make_shared<paraviewo::HDF5VTUWriter>();
		else
			tmpw = std::make_shared<paraviewo::VTUWriter>();
		paraviewo::ParaviewWriter &writer = *tmpw;

		if (opts.solve_export_to_file && opts.nodes)
			writer.add_field("nodes", node_fun);

		if (problem.is_time_dependent())
		{
			bool is_time_integrator_valid = time_integrator != nullptr;

			if (opts.velocity)
			{
				const Eigen::VectorXd velocity =
					is_time_integrator_valid ? (time_integrator->v_prev()) : Eigen::VectorXd::Zero(sol.size());
				save_volume_vector_field(state, points, opts, "velocity", velocity, writer);
			}

			if (opts.acceleration)
			{
				const Eigen::VectorXd acceleration =
					is_time_integrator_valid ? (time_integrator->a_prev()) : Eigen::VectorXd::Zero(sol.size());
				save_volume_vector_field(state, points, opts, "acceleration", acceleration, writer);
			}
		}

		if (opts.forces)
		{
			const double s = state.solve_data.time_integrator
								 ? state.solve_data.time_integrator->acceleration_scaling()
								 : 1;

			for (const auto &[name, form] : state.solve_data.named_forms())
			{
				// NOTE: Assumes this form will be null for the entire sim
				if (form == nullptr)
					continue;

				Eigen::VectorXd force;
				if (form->enabled())
				{
					form->first_derivative(sol, force);
					force *= -1.0 / s; // Divide by acceleration scaling to get units of force
				}
				else
				{
					force.setZero(sol.size());
				}

				save_volume_vector_field(state, points, opts, name + "_forces", force, writer);
			}
		}

		// if(problem->is_mixed())
		if (state.mixed_assembler != nullptr)
		{
			Eigen::MatrixXd interp_p;
			Evaluator::interpolate_function(
				mesh, 1, // FIXME: state.disc_orders should use pressure discr orders, works only with sampler
				pressure_bases, state.disc_orders, state.polys, state.polys_3d, ref_element_sampler,
				points.rows(), pressure, interp_p, opts.use_sampler, opts.boundary_only);

			if (obstacle.n_vertices() > 0)
			{
				interp_p.conservativeResize(interp_p.size() + obstacle.n_vertices(), 1);
				interp_p.bottomRows(obstacle.n_vertices()).setZero();
			}

			if (opts.solve_export_to_file)
				writer.add_field("pressure", interp_p);
			else
				solution_frames.back().pressure = interp_p;
		}

		if (obstacle.n_vertices() > 0)
		{
			discr.conservativeResize(discr.size() + obstacle.n_vertices(), 1);
			discr.bottomRows(obstacle.n_vertices()).setZero();
		}

		if (opts.solve_export_to_file && opts.discretization_order)
			writer.add_field("discr", discr);

		if (problem.has_exact_sol())
		{
			if (opts.solve_export_to_file)
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
			std::vector<assembler::Assembler::NamedMatrix> vals, tvals;
			Evaluator::compute_scalar_value(
				mesh, problem.is_scalar(), bases, gbases,
				state.disc_orders, state.polys, state.polys_3d,
				*state.assembler,
				ref_element_sampler, points.rows(), sol, t, vals, opts.use_sampler, opts.boundary_only);

			for (auto &[_, v] : vals)
				utils::append_rows_of_zeros(v, obstacle.n_vertices());

			if (opts.scalar_values)
			{
				if (opts.solve_export_to_file)
				{
					for (const auto &[name, v] : vals)
						writer.add_field(name, v);
				}
				else if (vals.size() > 0)
					solution_frames.back().scalar_value = vals[0].second;
			}

			if (opts.solve_export_to_file && opts.tensor_values)
			{
				Evaluator::compute_tensor_value(
					mesh, problem.is_scalar(), bases, gbases, state.disc_orders,
					state.polys, state.polys_3d, *state.assembler, ref_element_sampler,
					points.rows(), sol, t, tvals, opts.use_sampler, opts.boundary_only);

				for (auto &[_, v] : tvals)
					utils::append_rows_of_zeros(v, obstacle.n_vertices());

				for (const auto &[name, v] : tvals)
				{
					const int stride = mesh.dimension();
					assert(v.cols() % stride == 0);

					for (int i = 0; i < v.cols(); i += stride)
					{
						const Eigen::MatrixXd tmp = v.middleCols(i, stride);
						assert(tmp.cols() == stride);

						const int ii = (i / stride) + 1;
						writer.add_field(fmt::format("{:s}_{:d}", name, ii), tmp);
					}
				}
			}

			if (!opts.use_spline)
			{
				Evaluator::average_grad_based_function(
					mesh, problem.is_scalar(), state.n_bases, bases, gbases,
					state.disc_orders, state.polys, state.polys_3d, *state.assembler,
					ref_element_sampler, t, points.rows(), sol, vals, tvals,
					opts.use_sampler, opts.boundary_only);

				if (obstacle.n_vertices() > 0)
				{
					for (auto &v : vals)
					{
						v.second.conservativeResize(v.second.size() + obstacle.n_vertices(), 1);
						v.second.bottomRows(obstacle.n_vertices()).setZero();
					}
				}

				if (opts.scalar_values)
				{
					if (opts.solve_export_to_file)
					{
						for (const auto &v : vals)
							writer.add_field(fmt::format("{:s}_avg", v.first), v.second);
					}
					else if (vals.size() > 0)
						solution_frames.back().scalar_value_avg = vals[0].second;
				}
				// for(int i = 0; i < tvals.cols(); ++i){
				// 	const int ii = (i / mesh.dimension()) + 1;
				// 	const int jj = (i % mesh.dimension()) + 1;
				// 	writer.add_field("tensor_value_avg_" + std::to_string(ii) + std::to_string(jj), tvals.col(i));
				// }
			}
		}

		if (opts.material_params)
		{
			const auto &params = assembler.parameters();

			std::map<std::string, Eigen::MatrixXd> param_val;
			for (const auto &[p, _] : params)
				param_val[p] = Eigen::MatrixXd(points.rows(), 1);
			Eigen::MatrixXd rhos(points.rows(), 1);

			Eigen::MatrixXd local_pts;
			Eigen::MatrixXi vis_faces_poly, vis_edges_poly;

			int index = 0;
			const auto &sampler = ref_element_sampler;
			for (int e = 0; e < int(bases.size()); ++e)
			{
				const basis::ElementBases &gbs = gbases[e];
				const basis::ElementBases &bs = bases[e];

				if (opts.use_sampler)
				{
					if (mesh.is_simplex(e))
						local_pts = sampler.simplex_points();
					else if (mesh.is_cube(e))
						local_pts = sampler.cube_points();
					else
					{
						if (mesh.is_volume())
							sampler.sample_polyhedron(polys_3d.at(e).first, polys_3d.at(e).second, local_pts, vis_faces_poly, vis_edges_poly);
						else
							sampler.sample_polygon(polys.at(e), local_pts, vis_faces_poly, vis_edges_poly);
					}
				}
				else
				{
					if (mesh.is_volume())
					{
						if (mesh.is_simplex(e))
							autogen::p_nodes_3d(disc_orders(e), local_pts);
						else if (mesh.is_cube(e))
							autogen::q_nodes_3d(disc_orders(e), local_pts);
						else
							continue;
					}
					else
					{
						if (mesh.is_simplex(e))
							autogen::p_nodes_2d(disc_orders(e), local_pts);
						else if (mesh.is_cube(e))
							autogen::q_nodes_2d(disc_orders(e), local_pts);
						else
						{
							const auto &mesh2d = static_cast<const mesh::Mesh2D &>(mesh);
							const int n_v = mesh2d.n_face_vertices(e);
							local_pts.resize(n_v, 2);

							for (int j = 0; j < n_v; ++j)
							{
								local_pts.row(j) = mesh2d.point(mesh2d.face_vertex(e, j));
							}
						}
					}
				}

				assembler::ElementAssemblyValues vals;
				vals.compute(e, mesh.is_volume(), local_pts, bs, gbs);

				for (int j = 0; j < vals.val.rows(); ++j)
				{
					for (const auto &[p, func] : params)
						param_val.at(p)(index) = func(local_pts.row(j), vals.val.row(j), t, e);

					rhos(index) = density(local_pts.row(j), vals.val.row(j), t, e);

					++index;
				}
			}

			assert(index == points.rows());

			if (obstacle.n_vertices() > 0)
			{
				for (auto &[_, tmp] : param_val)
				{
					tmp.conservativeResize(tmp.size() + obstacle.n_vertices(), 1);
					tmp.bottomRows(obstacle.n_vertices()).setZero();
				}

				rhos.conservativeResize(rhos.size() + obstacle.n_vertices(), 1);
				rhos.bottomRows(obstacle.n_vertices()).setZero();
			}
			for (const auto &[p, tmp] : param_val)
				writer.add_field(p, tmp);
			writer.add_field("rho", rhos);
		}

		if (opts.body_ids)
		{

			Eigen::MatrixXd ids(points.rows(), 1);

			for (int i = 0; i < points.rows(); ++i)
			{
				ids(i) = mesh.get_body_id(el_id(i));
			}

			if (obstacle.n_vertices() > 0)
			{
				ids.conservativeResize(ids.size() + obstacle.n_vertices(), 1);
				ids.bottomRows(obstacle.n_vertices()).setZero();
			}

			writer.add_field("body_ids", ids);
		}

		// interpolate_function(pts_index, rhs, fun, opts.boundary_only);
		// writer.add_field("rhs", fun);

		if (fun.cols() != 1 && state.mixed_assembler == nullptr)
		{
			Eigen::MatrixXd traction_forces, traction_forces_fun;
			compute_traction_forces(state, sol, t, traction_forces, false);

			Evaluator::interpolate_function(
				mesh, problem.is_scalar(), bases, state.disc_orders,
				state.polys, state.polys_3d, ref_element_sampler,
				points.rows(), traction_forces, traction_forces_fun, opts.use_sampler, opts.boundary_only);

			if (obstacle.n_vertices() > 0)
			{
				traction_forces_fun.conservativeResize(traction_forces_fun.rows() + obstacle.n_vertices(), traction_forces_fun.cols());
				traction_forces_fun.bottomRows(obstacle.n_vertices()).setZero();
			}

			writer.add_field("traction_force", traction_forces_fun);
		}

		if (fun.cols() != 1 && state.mixed_assembler == nullptr)
		{
			try
			{
				Eigen::MatrixXd potential_grad, potential_grad_fun;
				state.assembler->assemble_gradient(mesh.is_volume(), state.n_bases, bases, gbases, state.ass_vals_cache, t, dt, sol, sol, potential_grad);

				Evaluator::interpolate_function(
					mesh, problem.is_scalar(), bases, state.disc_orders,
					state.polys, state.polys_3d, ref_element_sampler,
					points.rows(), potential_grad, potential_grad_fun, opts.use_sampler, opts.boundary_only);

				if (obstacle.n_vertices() > 0)
				{
					potential_grad_fun.conservativeResize(potential_grad_fun.rows() + obstacle.n_vertices(), potential_grad_fun.cols());
					potential_grad_fun.bottomRows(obstacle.n_vertices()).setZero();
				}

				writer.add_field("gradient_of_potential", potential_grad_fun);
			}
			catch (std::exception &)
			{
			}
		}

		// Write the solution last so it is the default for warp-by-vector
		if (opts.solve_export_to_file)
			writer.add_field("solution", fun);
		else
			solution_frames.back().solution = fun;

		if (opts.solve_export_to_file)
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
				writer.write_mesh(path, points, elements, true, disc_orders.maxCoeff() == 1);
		}
		else
		{
			solution_frames.back().name = path;
			solution_frames.back().points = points;
			solution_frames.back().connectivity = tets;
		}
	}

	void OutGeometryData::save_volume_vector_field(
		const State &state,
		const Eigen::MatrixXd &points,
		const ExportOptions &opts,
		const std::string &name,
		const Eigen::VectorXd &field,
		paraviewo::ParaviewWriter &writer) const
	{
		Eigen::MatrixXd inerpolated_field;
		Evaluator::interpolate_function(
			*state.mesh, state.problem->is_scalar(), state.bases, state.disc_orders,
			state.polys, state.polys_3d, ref_element_sampler,
			points.rows(), field, inerpolated_field, opts.use_sampler, opts.boundary_only);

		if (state.obstacle.n_vertices() > 0)
		{
			inerpolated_field.conservativeResize(
				inerpolated_field.rows() + state.obstacle.n_vertices(), inerpolated_field.cols());
			inerpolated_field.bottomRows(state.obstacle.n_vertices()) =
				utils::unflatten(field.tail(state.obstacle.ndof()), inerpolated_field.cols());
		}

		if (opts.solve_export_to_file)
		{
			writer.add_field(name, inerpolated_field);
		}
		// TODO: else save to solution frames
	}

	void OutGeometryData::save_surface(
		const std::string &export_surface,
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &pressure,
		const double t,
		const double dt_in,
		const ExportOptions &opts,
		const bool is_contact_enabled,
		std::vector<SolutionFrame> &solution_frames) const
	{

		const Eigen::VectorXi &disc_orders = state.disc_orders;
		const auto &density = state.mass_matrix_assembler->density();
		const std::vector<basis::ElementBases> &bases = state.bases;
		const std::vector<basis::ElementBases> &pressure_bases = state.pressure_bases;
		const std::vector<basis::ElementBases> &gbases = state.geom_bases();
		const assembler::Assembler &assembler = *state.assembler;
		const assembler::Problem &problem = *state.problem;
		const mesh::Mesh &mesh = *state.mesh;
		int problem_dim = (problem.is_scalar() ? 1 : mesh.dimension());

		Eigen::MatrixXd boundary_vis_vertices;
		Eigen::MatrixXd boundary_vis_local_vertices;
		Eigen::MatrixXi boundary_vis_elements;
		Eigen::MatrixXi boundary_vis_elements_ids;
		Eigen::MatrixXi boundary_vis_primitive_ids;
		Eigen::MatrixXd boundary_vis_normals;
		Eigen::MatrixXd displaced_boundary_vis_normals;

		build_vis_boundary_mesh(mesh, bases, gbases, state.total_local_boundary, sol, problem_dim,
								boundary_vis_vertices, boundary_vis_local_vertices, boundary_vis_elements,
								boundary_vis_elements_ids, boundary_vis_primitive_ids, boundary_vis_normals,
								displaced_boundary_vis_normals);

		Eigen::MatrixXd fun, interp_p, discr, vect, b_sidesets;

		Eigen::MatrixXd lsol, lp, lgrad, lpgrad;

		int actual_dim = 1;
		if (!problem.is_scalar())
			actual_dim = mesh.dimension();

		discr.resize(boundary_vis_vertices.rows(), 1);
		fun.resize(boundary_vis_vertices.rows(), actual_dim);
		interp_p.resize(boundary_vis_vertices.rows(), 1);
		vect.resize(boundary_vis_vertices.rows(), mesh.dimension());

		b_sidesets.resize(boundary_vis_vertices.rows(), 1);
		b_sidesets.setZero();

		for (int i = 0; i < boundary_vis_vertices.rows(); ++i)
		{
			const auto s_id = mesh.get_boundary_id(boundary_vis_primitive_ids(i));
			if (s_id > 0)
			{
				b_sidesets(i) = s_id;
			}

			const int el_index = boundary_vis_elements_ids(i);
			Evaluator::interpolate_at_local_vals(
				mesh, problem.is_scalar(), bases, gbases,
				el_index, boundary_vis_local_vertices.row(i), sol, lsol, lgrad);
			assert(lsol.size() == actual_dim);
			if (state.mixed_assembler != nullptr)
			{
				Evaluator::interpolate_at_local_vals(
					mesh, 1, pressure_bases, gbases,
					el_index, boundary_vis_local_vertices.row(i), pressure, lp, lpgrad);
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
				assert(lgrad.size() == mesh.dimension());
				for (int j = 0; j < mesh.dimension(); ++j)
				{
					vect(i, j) = lgrad(j);
				}
			}
			else
			{
				assert(lgrad.size() == actual_dim * actual_dim);
				std::vector<assembler::Assembler::NamedMatrix> tensor_flat;
				const basis::ElementBases &gbs = gbases[el_index];
				const basis::ElementBases &bs = bases[el_index];
				assembler.compute_tensor_value(assembler::OutputData(t, el_index, bs, gbs, boundary_vis_local_vertices.row(i), sol), tensor_flat);
				// TF computed only from cauchy stress
				assert(tensor_flat[0].first == "cauchy_stess");
				assert(tensor_flat[0].second.size() == actual_dim * actual_dim);

				Eigen::Map<Eigen::MatrixXd> tensor(tensor_flat[0].second.data(), actual_dim, actual_dim);
				vect.row(i) = displaced_boundary_vis_normals.row(i) * tensor;

				double area = 0;
				if (mesh.is_volume())
				{
					if (mesh.is_simplex(el_index))
						area = mesh.tri_area(boundary_vis_primitive_ids(i));
					else if (mesh.is_cube(el_index))
						area = mesh.quad_area(boundary_vis_primitive_ids(i));
				}
				else
					area = mesh.edge_length(boundary_vis_primitive_ids(i));

				vect.row(i) *= area;
			}
		}

		std::shared_ptr<paraviewo::ParaviewWriter> tmpw;
		if (opts.use_hdf5)
			tmpw = std::make_shared<paraviewo::HDF5VTUWriter>();
		else
			tmpw = std::make_shared<paraviewo::VTUWriter>();
		paraviewo::ParaviewWriter &writer = *tmpw;

		if (opts.solve_export_to_file)
		{

			writer.add_field("normals", boundary_vis_normals);
			writer.add_field("displaced_normals", displaced_boundary_vis_normals);
			if (state.mixed_assembler != nullptr)
				writer.add_field("pressure", interp_p);
			writer.add_field("discr", discr);
			writer.add_field("sidesets", b_sidesets);

			if (actual_dim == 1)
				writer.add_field("solution_grad", vect);
			else
			{
				writer.add_field("traction_force", vect);
			}
		}
		else
		{
			if (state.mixed_assembler != nullptr)
				solution_frames.back().pressure = interp_p;
		}

		if (opts.material_params)
		{
			const auto &params = assembler.parameters();

			std::map<std::string, Eigen::MatrixXd> param_val;
			for (const auto &[p, _] : params)
				param_val[p] = Eigen::MatrixXd(boundary_vis_vertices.rows(), 1);
			Eigen::MatrixXd rhos(boundary_vis_vertices.rows(), 1);

			for (int i = 0; i < boundary_vis_vertices.rows(); ++i)
			{
				double lambda, mu;

				for (const auto &[p, func] : params)
					param_val.at(p)(i) = func(boundary_vis_local_vertices.row(i), boundary_vis_vertices.row(i), t, boundary_vis_elements_ids(i));

				rhos(i) = density(boundary_vis_local_vertices.row(i), boundary_vis_vertices.row(i), t, boundary_vis_elements_ids(i));
			}

			for (const auto &[p, tmp] : param_val)
				writer.add_field(p, tmp);
			writer.add_field("rho", rhos);
		}

		if (opts.body_ids)
		{

			Eigen::MatrixXd ids(boundary_vis_vertices.rows(), 1);

			for (int i = 0; i < boundary_vis_vertices.rows(); ++i)
			{
				ids(i) = mesh.get_body_id(boundary_vis_elements_ids(i));
			}

			writer.add_field("body_ids", ids);
		}

		// Write the solution last so it is the default for warp-by-vector
		if (opts.solve_export_to_file)
			writer.add_field("solution", fun);
		else
			solution_frames.back().solution = fun;

		if (opts.solve_export_to_file)
			writer.write_mesh(export_surface, boundary_vis_vertices, boundary_vis_elements);
		else
		{
			solution_frames.back().name = export_surface;
			solution_frames.back().points = boundary_vis_vertices;
			solution_frames.back().connectivity = boundary_vis_elements;
		}
	}

	void OutGeometryData::save_contact_surface(
		const std::string &export_surface,
		const State &state,
		const Eigen::MatrixXd &sol,
		const Eigen::MatrixXd &pressure,
		const double t,
		const double dt_in,
		const ExportOptions &opts,
		const bool is_contact_enabled,
		std::vector<SolutionFrame> &solution_frames) const
	{
		const mesh::Mesh &mesh = *state.mesh;
		const ipc::CollisionMesh &collision_mesh = state.collision_mesh;
		const double dhat = state.args["contact"]["dhat"];
		const double friction_coefficient = state.args["contact"]["friction_coefficient"];

		double global_static_friction_coefficient = state.args["contact"].contains("static_friction_coefficient") ? state.args["contact"]["static_friction_coefficient"]["global"].get<double>()  : friction_coefficient;
		double global_kinetic_friction_coefficient = state.args["contact"].contains("kinetic_friction_coefficient") ? state.args["contact"]["kinetic_friction_coefficient"]["global"].get<double>()  : friction_coefficient;
		double static_friction_coefficient = global_static_friction_coefficient;
		double kinetic_friction_coefficient = global_kinetic_friction_coefficient;

		std::map<std::tuple<int, int>, std::pair<double, double>> pairwise_friction_;

		// Handle pairwise static friction coefficients
		if (state.args["contact"].contains("static_friction_coefficient") &&
			state.args["contact"]["static_friction_coefficient"].contains("pairwise"))
		{
			const auto &pairwise_static = state.args["contact"]["static_friction_coefficient"]["pairwise"];
			for (const auto &pair : pairwise_static)
			{
				const auto &ids = pair["ids"];
				int id0 = (ids[0] == "all") ? -1 : ids[0].get<int>();
				int id1 = (ids[1] == "all") ? -1 : ids[1].get<int>();
				double static_friction_value = pair["value"];

				pairwise_friction_[std::make_tuple(id0, id1)].first = static_friction_value;
			}
		}

		// Handle pairwise kinetic friction coefficients
		if (state.args["contact"].contains("kinetic_friction_coefficient") &&
			state.args["contact"]["kinetic_friction_coefficient"].contains("pairwise"))
		{
			const auto &pairwise_kinetic = state.args["contact"]["kinetic_friction_coefficient"]["pairwise"];
			for (const auto &pair : pairwise_kinetic)
			{
				const auto &ids = pair["ids"];
				int id0 = (ids[0] == "all") ? -1 : ids[0].get<int>();
				int id1 = (ids[1] == "all") ? -1 : ids[1].get<int>();
				double kinetic_friction_value = pair["value"];

				// Update or insert the kinetic friction value
				pairwise_friction_[std::make_tuple(id0, id1)].second = kinetic_friction_value;
			}
		}

		const double epsv = state.args["contact"]["epsv"];
		const std::shared_ptr<solver::ContactForm> &contact_form = state.solve_data.contact_form;
		const std::shared_ptr<solver::FrictionForm> &friction_form = state.solve_data.friction_form;

		if (opts.solve_export_to_file)
		{
			std::shared_ptr<paraviewo::ParaviewWriter> tmpw;
			if (opts.use_hdf5)
				tmpw = std::make_shared<paraviewo::HDF5VTUWriter>();
			else
				tmpw = std::make_shared<paraviewo::VTUWriter>();
			paraviewo::ParaviewWriter &writer = *tmpw;

			const int problem_dim = mesh.dimension();
			const Eigen::MatrixXd full_displacements = utils::unflatten(sol, problem_dim);
			const Eigen::MatrixXd surface_displacements = collision_mesh.map_displacements(full_displacements);

			const Eigen::MatrixXd displaced_surface = collision_mesh.displace_vertices(full_displacements);

			ipc::Collisions collision_set;
			collision_set.set_use_area_weighting(state.args["contact"]["use_convergent_formulation"]);
			collision_set.build(
				collision_mesh, displaced_surface, dhat,
				/*dmin=*/0, state.args["solver"]["contact"]["CCD"]["broad_phase"]);

			ipc::BarrierPotential barrier_potential(dhat);

			const double barrier_stiffness = contact_form != nullptr ? contact_form->barrier_stiffness() : 1;

			if (opts.contact_forces)
			{
				Eigen::MatrixXd forces = -barrier_stiffness * barrier_potential.gradient(collision_set, collision_mesh, displaced_surface);

				Eigen::MatrixXd forces_reshaped = utils::unflatten(forces, problem_dim);

				assert(forces_reshaped.rows() == surface_displacements.rows());
				assert(forces_reshaped.cols() == surface_displacements.cols());
				writer.add_field("contact_forces", forces_reshaped);
			}

			if (opts.friction_forces)
			{

				ipc::FrictionCollisions friction_collision_set;
				friction_collision_set.build(
					collision_mesh, displaced_surface, collision_set,
					barrier_potential, barrier_stiffness, friction_coefficient,
					static_friction_coefficient, kinetic_friction_coefficient, pairwise_friction_);

				ipc::FrictionPotential friction_potential(epsv);

				Eigen::MatrixXd velocities;
				if (state.solve_data.time_integrator != nullptr)
					velocities = state.solve_data.time_integrator->v_prev();
				else
					velocities = sol;
				velocities = collision_mesh.map_displacements(utils::unflatten(velocities, collision_mesh.dim()));

				Eigen::MatrixXd forces = -friction_potential.gradient(
					friction_collision_set, collision_mesh, velocities);

				Eigen::MatrixXd forces_reshaped = utils::unflatten(forces, problem_dim);

				assert(forces_reshaped.rows() == surface_displacements.rows());
				assert(forces_reshaped.cols() == surface_displacements.cols());
				writer.add_field("friction_forces", forces_reshaped);
			}

			assert(collision_mesh.rest_positions().rows() == surface_displacements.rows());
			assert(collision_mesh.rest_positions().cols() == surface_displacements.cols());

			// Write the solution last so it is the default for warp-by-vector
			writer.add_field("solution", surface_displacements);

			writer.write_mesh(
				export_surface.substr(0, export_surface.length() - 4) + "_contact.vtu",
				collision_mesh.rest_positions(),
				problem_dim == 3 ? collision_mesh.faces() : collision_mesh.edges());
		}
	}

	void OutGeometryData::save_wire(
		const std::string &name,
		const State &state,
		const Eigen::MatrixXd &sol,
		const double t,
		const ExportOptions &opts,
		std::vector<SolutionFrame> &solution_frames) const
	{
		const std::vector<basis::ElementBases> &gbases = state.geom_bases();
		const mesh::Mesh &mesh = *state.mesh;
		const assembler::Problem &problem = *state.problem;

		if (!opts.solve_export_to_file) // TODO?
			return;
		const auto &sampler = ref_element_sampler;

		Eigen::MatrixXi vis_faces_poly, vis_edges_poly;
		Eigen::MatrixXd vis_pts_poly;

		const auto &current_bases = gbases;
		int seg_total_size = 0;
		int pts_total_size = 0;
		int faces_total_size = 0;

		for (size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if (mesh.is_simplex(i))
			{
				pts_total_size += sampler.simplex_points().rows();
				seg_total_size += sampler.simplex_edges().rows();
				faces_total_size += sampler.simplex_faces().rows();
			}
			else if (mesh.is_cube(i))
			{
				pts_total_size += sampler.cube_points().rows();
				seg_total_size += sampler.cube_edges().rows();
				faces_total_size += sampler.cube_faces().rows();
			}
			else
			{
				if (mesh.is_volume())
					sampler.sample_polyhedron(state.polys_3d.at(i).first, state.polys_3d.at(i).second, vis_pts_poly, vis_faces_poly, vis_edges_poly);
				else
					sampler.sample_polygon(state.polys.at(i), vis_pts_poly, vis_faces_poly, vis_edges_poly);

				pts_total_size += vis_pts_poly.rows();
				seg_total_size += vis_edges_poly.rows();
				faces_total_size += vis_faces_poly.rows();
			}
		}

		Eigen::MatrixXd points(pts_total_size, mesh.dimension());
		Eigen::MatrixXi edges(seg_total_size, 2);
		Eigen::MatrixXi faces(faces_total_size, 3);
		points.setZero();

		Eigen::MatrixXd mapped, tmp;
		int seg_index = 0, pts_index = 0, face_index = 0;
		for (size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if (mesh.is_simplex(i))
			{
				bs.eval_geom_mapping(sampler.simplex_points(), mapped);
				edges.block(seg_index, 0, sampler.simplex_edges().rows(), edges.cols()) = sampler.simplex_edges().array() + pts_index;
				seg_index += sampler.simplex_edges().rows();

				faces.block(face_index, 0, sampler.simplex_faces().rows(), 3) = sampler.simplex_faces().array() + pts_index;
				face_index += sampler.simplex_faces().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				pts_index += mapped.rows();
			}
			else if (mesh.is_cube(i))
			{
				bs.eval_geom_mapping(sampler.cube_points(), mapped);
				edges.block(seg_index, 0, sampler.cube_edges().rows(), edges.cols()) = sampler.cube_edges().array() + pts_index;
				seg_index += sampler.cube_edges().rows();

				faces.block(face_index, 0, sampler.cube_faces().rows(), 3) = sampler.cube_faces().array() + pts_index;
				face_index += sampler.cube_faces().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				pts_index += mapped.rows();
			}
			else
			{
				if (mesh.is_volume())
					sampler.sample_polyhedron(state.polys_3d.at(i).first, state.polys_3d.at(i).second, vis_pts_poly, vis_faces_poly, vis_edges_poly);
				else
					sampler.sample_polygon(state.polys.at(i), vis_pts_poly, vis_faces_poly, vis_edges_poly);

				edges.block(seg_index, 0, vis_edges_poly.rows(), edges.cols()) = vis_edges_poly.array() + pts_index;
				seg_index += vis_edges_poly.rows();

				faces.block(face_index, 0, vis_faces_poly.rows(), 3) = vis_faces_poly.array() + pts_index;
				face_index += vis_faces_poly.rows();

				points.block(pts_index, 0, vis_pts_poly.rows(), points.cols()) = vis_pts_poly;
				pts_index += vis_pts_poly.rows();
			}
		}

		assert(pts_index == points.rows());
		assert(face_index == faces.rows());

		if (mesh.is_volume())
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
			Eigen::Matrix2d mmat;
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
		Evaluator::interpolate_function(
			mesh, problem.is_scalar(), state.bases, state.disc_orders,
			state.polys, state.polys_3d, ref_element_sampler,
			pts_index, sol, fun, /*use_sampler*/ true, false);

		Eigen::MatrixXd exact_fun, err;

		if (problem.has_exact_sol())
		{
			problem.exact(points, t, exact_fun);
			err = (fun - exact_fun).eval().rowwise().norm();
		}

		std::shared_ptr<paraviewo::ParaviewWriter> tmpw;
		if (opts.use_hdf5)
			tmpw = std::make_shared<paraviewo::HDF5VTUWriter>();
		else
			tmpw = std::make_shared<paraviewo::VTUWriter>();
		paraviewo::ParaviewWriter &writer = *tmpw;

		if (problem.has_exact_sol())
		{
			writer.add_field("exact", exact_fun);
			writer.add_field("error", err);
		}

		if (fun.cols() != 1)
		{
			std::vector<assembler::Assembler::NamedMatrix> scalar_val;
			Evaluator::compute_scalar_value(
				mesh, problem.is_scalar(), state.bases, gbases,
				state.disc_orders, state.polys, state.polys_3d,
				*state.assembler,
				ref_element_sampler, pts_index, sol, t, scalar_val, /*use_sampler*/ true, false);
			for (const auto &v : scalar_val)
				writer.add_field(v.first, v.second);
		}
		// Write the solution last so it is the default for warp-by-vector
		writer.add_field("solution", fun);

		writer.write_mesh(name, points, edges);
	}

	void OutGeometryData::save_points(
		const std::string &path,
		const State &state,
		const Eigen::MatrixXd &sol,
		const ExportOptions &opts,
		std::vector<SolutionFrame> &solution_frames) const
	{
		const auto &dirichlet_nodes = state.dirichlet_nodes;
		const auto &dirichlet_nodes_position = state.dirichlet_nodes_position;
		const mesh::Mesh &mesh = *state.mesh;
		const assembler::Problem &problem = *state.problem;

		int actual_dim = 1;
		if (!problem.is_scalar())
			actual_dim = mesh.dimension();

		Eigen::MatrixXd fun(dirichlet_nodes_position.size(), actual_dim);
		Eigen::MatrixXd b_sidesets(dirichlet_nodes_position.size(), 1);
		b_sidesets.setZero();
		Eigen::MatrixXd points(dirichlet_nodes_position.size(), mesh.dimension());
		std::vector<std::vector<int>> cells(dirichlet_nodes_position.size());

		for (int i = 0; i < dirichlet_nodes_position.size(); ++i)
		{
			const int n_id = dirichlet_nodes[i];
			const auto s_id = mesh.get_node_id(n_id);
			if (s_id > 0)
			{
				b_sidesets(i) = s_id;
			}

			for (int j = 0; j < actual_dim; ++j)
			{
				fun(i, j) = sol(n_id * actual_dim + j);
			}

			points.row(i) = dirichlet_nodes_position[i];
			cells[i].push_back(i);
		}

		std::shared_ptr<paraviewo::ParaviewWriter> tmpw;
		if (opts.use_hdf5)
			tmpw = std::make_shared<paraviewo::HDF5VTUWriter>();
		else
			tmpw = std::make_shared<paraviewo::VTUWriter>();
		paraviewo::ParaviewWriter &writer = *tmpw;

		if (opts.solve_export_to_file)
		{
			writer.add_field("sidesets", b_sidesets);
			// Write the solution last so it is the default for warp-by-vector
			writer.add_field("solution", fun);
			writer.write_mesh(path, points, cells, false, false);
		}
	}

	void OutGeometryData::save_pvd(
		const std::string &name,
		const std::function<std::string(int)> &vtu_names,
		int time_steps, double t0, double dt, int skip_frame) const
	{
		paraviewo::PVDWriter::save_pvd(name, vtu_names, time_steps, t0, dt, skip_frame);
	}

	void OutGeometryData::init_sampler(const polyfem::mesh::Mesh &mesh, const double vismesh_rel_area)
	{
		ref_element_sampler.init(mesh.is_volume(), mesh.n_elements(), vismesh_rel_area);
	}

	void OutGeometryData::build_grid(const polyfem::mesh::Mesh &mesh, const double spacing)
	{
		if (spacing <= 0)
			return;

		RowVectorNd min, max;
		mesh.bounding_box(min, max);
		const RowVectorNd delta = max - min;
		const int nx = delta[0] / spacing + 1;
		const int ny = delta[1] / spacing + 1;
		const int nz = delta.cols() >= 3 ? (delta[2] / spacing + 1) : 1;
		const int n = nx * ny * nz;

		grid_points.resize(n, delta.cols());
		int index = 0;
		for (int i = 0; i < nx; ++i)
		{
			const double x = (delta[0] / (nx - 1)) * i + min[0];

			for (int j = 0; j < ny; ++j)
			{
				const double y = (delta[1] / (ny - 1)) * j + min[1];

				if (delta.cols() <= 2)
				{
					grid_points.row(index++) << x, y;
				}
				else
				{
					for (int k = 0; k < nz; ++k)
					{
						const double z = (delta[2] / (nz - 1)) * k + min[2];
						grid_points.row(index++) << x, y, z;
					}
				}
			}
		}

		assert(index == n);

		std::vector<std::array<Eigen::Vector3d, 2>> boxes;
		mesh.elements_boxes(boxes);

		SimpleBVH::BVH bvh;
		bvh.init(boxes);

		const double eps = 1e-6;

		grid_points_to_elements.resize(grid_points.rows(), 1);
		grid_points_to_elements.setConstant(-1);

		grid_points_bc.resize(grid_points.rows(), mesh.is_volume() ? 4 : 3);

		for (int i = 0; i < grid_points.rows(); ++i)
		{
			const Eigen::Vector3d min(
				grid_points(i, 0) - eps,
				grid_points(i, 1) - eps,
				(mesh.is_volume() ? grid_points(i, 2) : 0) - eps);

			const Eigen::Vector3d max(
				grid_points(i, 0) + eps,
				grid_points(i, 1) + eps,
				(mesh.is_volume() ? grid_points(i, 2) : 0) + eps);

			std::vector<unsigned int> candidates;

			bvh.intersect_box(min, max, candidates);

			for (const auto cand : candidates)
			{
				if (!mesh.is_simplex(cand))
				{
					logger().warn("Element {} is not simplex, skipping", cand);
					continue;
				}

				Eigen::MatrixXd coords;
				mesh.barycentric_coords(grid_points.row(i), cand, coords);

				for (int d = 0; d < coords.size(); ++d)
				{
					if (fabs(coords(d)) < 1e-8)
						coords(d) = 0;
					else if (fabs(coords(d) - 1) < 1e-8)
						coords(d) = 1;
				}

				if (coords.array().minCoeff() >= 0 && coords.array().maxCoeff() <= 1)
				{
					grid_points_to_elements(i) = cand;
					grid_points_bc.row(i) = coords;
					break;
				}
			}
		}
	}

	void OutStatsData::compute_mesh_size(const polyfem::mesh::Mesh &mesh_in, const std::vector<polyfem::basis::ElementBases> &bases_in, const int n_samples, const bool use_curved_mesh_size)
	{
		Eigen::MatrixXd samples_simplex, samples_cube, mapped, p0, p1, p;

		mesh_size = 0;
		average_edge_length = 0;
		min_edge_length = std::numeric_limits<double>::max();

		if (!use_curved_mesh_size)
		{
			mesh_in.get_edges(p0, p1);
			p = p0 - p1;
			min_edge_length = p.rowwise().norm().minCoeff();
			average_edge_length = p.rowwise().norm().mean();
			mesh_size = p.rowwise().norm().maxCoeff();

			logger().info("hmin: {}", min_edge_length);
			logger().info("hmax: {}", mesh_size);
			logger().info("havg: {}", average_edge_length);

			return;
		}

		if (mesh_in.is_volume())
		{
			utils::EdgeSampler::sample_3d_simplex(n_samples, samples_simplex);
			utils::EdgeSampler::sample_3d_cube(n_samples, samples_cube);
		}
		else
		{
			utils::EdgeSampler::sample_2d_simplex(n_samples, samples_simplex);
			utils::EdgeSampler::sample_2d_cube(n_samples, samples_cube);
		}

		int n = 0;
		for (size_t i = 0; i < bases_in.size(); ++i)
		{
			if (mesh_in.is_polytope(i))
				continue;
			int n_edges;

			if (mesh_in.is_simplex(i))
			{
				n_edges = mesh_in.is_volume() ? 6 : 3;
				bases_in[i].eval_geom_mapping(samples_simplex, mapped);
			}
			else
			{
				n_edges = mesh_in.is_volume() ? 12 : 4;
				bases_in[i].eval_geom_mapping(samples_cube, mapped);
			}

			for (int j = 0; j < n_edges; ++j)
			{
				double current_edge = 0;
				for (int k = 0; k < n_samples - 1; ++k)
				{
					p0 = mapped.row(j * n_samples + k);
					p1 = mapped.row(j * n_samples + k + 1);
					p = p0 - p1;

					current_edge += p.norm();
				}

				mesh_size = std::max(current_edge, mesh_size);
				min_edge_length = std::min(current_edge, min_edge_length);
				average_edge_length += current_edge;
				++n;
			}
		}

		average_edge_length /= n;

		logger().info("hmin: {}", min_edge_length);
		logger().info("hmax: {}", mesh_size);
		logger().info("havg: {}", average_edge_length);
	}

	void OutStatsData::reset()
	{
		sigma_avg = 0;
		sigma_max = 0;
		sigma_min = 0;

		n_flipped = 0;
	}

	void OutStatsData::count_flipped_elements(const polyfem::mesh::Mesh &mesh, const std::vector<polyfem::basis::ElementBases> &gbases)
	{
		using namespace mesh;

		logger().info("Counting flipped elements...");
		const auto &els_tag = mesh.elements_tag();

		// flipped_elements.clear();
		for (size_t i = 0; i < gbases.size(); ++i)
		{
			if (mesh.is_polytope(i))
				continue;

			polyfem::assembler::ElementAssemblyValues vals;
			if (!vals.is_geom_mapping_positive(mesh.is_volume(), gbases[i]))
			{
				++n_flipped;

				static const std::vector<std::string> element_type_names{{
					"Simplex",
					"RegularInteriorCube",
					"RegularBoundaryCube",
					"SimpleSingularInteriorCube",
					"MultiSingularInteriorCube",
					"SimpleSingularBoundaryCube",
					"InterfaceCube",
					"MultiSingularBoundaryCube",
					"BoundaryPolytope",
					"InteriorPolytope",
					"Undefined",
				}};

				log_and_throw_error("element {} is flipped, type {}", i, element_type_names[static_cast<int>(els_tag[i])]);
			}
		}

		logger().info(" done");

		// dynamic_cast<Mesh3D *>(mesh.get())->save({56}, 1, "mesh.HYBRID");

		// std::sort(flipped_elements.begin(), flipped_elements.end());
		// auto it = std::unique(flipped_elements.begin(), flipped_elements.end());
		// flipped_elements.resize(std::distance(flipped_elements.begin(), it));
	}

	void OutStatsData::compute_errors(
		const int n_bases,
		const std::vector<polyfem::basis::ElementBases> &bases,
		const std::vector<polyfem::basis::ElementBases> &gbases,
		const polyfem::mesh::Mesh &mesh,
		const assembler::Problem &problem,
		const double tend,
		const Eigen::MatrixXd &sol)
	{
		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}
		if (sol.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		int actual_dim = 1;
		if (!problem.is_scalar())
			actual_dim = mesh.dimension();

		igl::Timer timer;
		timer.start();
		logger().info("Computing errors...");
		using std::max;

		const int n_el = int(bases.size());

		Eigen::MatrixXd v_exact, v_approx;
		Eigen::MatrixXd v_exact_grad(0, 0), v_approx_grad;

		l2_err = 0;
		h1_err = 0;
		grad_max_err = 0;
		h1_semi_err = 0;
		linf_err = 0;
		lp_err = 0;
		// double pred_norm = 0;

		static const int p = 8;

		// Eigen::MatrixXd err_per_el(n_el, 5);
		polyfem::assembler::ElementAssemblyValues vals;

		for (int e = 0; e < n_el; ++e)
		{
			vals.compute(e, mesh.is_volume(), bases[e], gbases[e]);

			if (problem.has_exact_sol())
			{
				problem.exact(vals.val, tend, v_exact);
				problem.exact_grad(vals.val, tend, v_exact_grad);
			}

			v_approx.resize(vals.val.rows(), actual_dim);
			v_approx.setZero();

			v_approx_grad.resize(vals.val.rows(), mesh.dimension() * actual_dim);
			v_approx_grad.setZero();

			const int n_loc_bases = int(vals.basis_values.size());

			for (int i = 0; i < n_loc_bases; ++i)
			{
				const auto &val = vals.basis_values[i];

				for (size_t ii = 0; ii < val.global.size(); ++ii)
				{
					for (int d = 0; d < actual_dim; ++d)
					{
						v_approx.col(d) += val.global[ii].val * sol(val.global[ii].index * actual_dim + d) * val.val;
						v_approx_grad.block(0, d * val.grad_t_m.cols(), v_approx_grad.rows(), val.grad_t_m.cols()) += val.global[ii].val * sol(val.global[ii].index * actual_dim + d) * val.grad_t_m;
					}
				}
			}

			const auto err = problem.has_exact_sol() ? (v_exact - v_approx).eval().rowwise().norm().eval() : (v_approx).eval().rowwise().norm().eval();
			const auto err_grad = problem.has_exact_sol() ? (v_exact_grad - v_approx_grad).eval().rowwise().norm().eval() : (v_approx_grad).eval().rowwise().norm().eval();

			// for(long i = 0; i < err.size(); ++i)
			// errors.push_back(err(i));

			linf_err = std::max(linf_err, err.maxCoeff());
			grad_max_err = std::max(linf_err, err_grad.maxCoeff());

			// {
			// 	const auto &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());
			// 	const auto v0 = mesh3d.point(mesh3d.cell_vertex(e, 0));
			// 	const auto v1 = mesh3d.point(mesh3d.cell_vertex(e, 1));
			// 	const auto v2 = mesh3d.point(mesh3d.cell_vertex(e, 2));
			// 	const auto v3 = mesh3d.point(mesh3d.cell_vertex(e, 3));

			// 	Eigen::Matrix<double, 6, 3> ee;
			// 	ee.row(0) = v0 - v1;
			// 	ee.row(1) = v1 - v2;
			// 	ee.row(2) = v2 - v0;

			// 	ee.row(3) = v0 - v3;
			// 	ee.row(4) = v1 - v3;
			// 	ee.row(5) = v2 - v3;

			// 	Eigen::Matrix<double, 6, 1> en = ee.rowwise().norm();

			// 	// Eigen::Matrix<double, 3*4, 1> alpha;
			// 	// alpha(0) = angle3(e.row(0), -e.row(1));	 	alpha(1) = angle3(e.row(1), -e.row(2));	 	alpha(2) = angle3(e.row(2), -e.row(0));
			// 	// alpha(3) = angle3(e.row(0), -e.row(4));	 	alpha(4) = angle3(e.row(4), e.row(3));	 	alpha(5) = angle3(-e.row(3), -e.row(0));
			// 	// alpha(6) = angle3(-e.row(4), -e.row(1));	alpha(7) = angle3(e.row(1), -e.row(5));	 	alpha(8) = angle3(e.row(5), e.row(4));
			// 	// alpha(9) = angle3(-e.row(2), -e.row(5));	alpha(10) = angle3(e.row(5), e.row(3));		alpha(11) = angle3(-e.row(3), e.row(2));

			// 	const double S = (ee.row(0).cross(ee.row(1)).norm() + ee.row(0).cross(ee.row(4)).norm() + ee.row(4).cross(ee.row(1)).norm() + ee.row(2).cross(ee.row(5)).norm()) / 2;
			// 	const double V = std::abs(ee.row(3).dot(ee.row(2).cross(-ee.row(0))))/6;
			// 	const double rho = 3 * V / S;
			// 	const double hp = en.maxCoeff();
			// 	const int pp = disc_orders(e);
			// 	const int p_ref = args["space"]["discr_order"];

			// 	err_per_el(e, 0) = err.mean();
			// 	err_per_el(e, 1) = err.maxCoeff();
			// 	err_per_el(e, 2) = std::pow(hp, pp+1)/(rho/hp); // /std::pow(average_edge_length, p_ref+1) * (sqrt(6)/12);
			// 	err_per_el(e, 3) = rho/hp;
			// 	err_per_el(e, 4) = (vals.det.array() * vals.quadrature.weights.array()).sum();

			// 	// pred_norm += (pow(std::pow(hp, pp+1)/(rho/hp),p) * vals.det.array() * vals.quadrature.weights.array()).sum();
			// }

			l2_err += (err.array() * err.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			h1_err += (err_grad.array() * err_grad.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			lp_err += (err.array().pow(p) * vals.det.array() * vals.quadrature.weights.array()).sum();
		}

		h1_semi_err = sqrt(fabs(h1_err));
		h1_err = sqrt(fabs(l2_err) + fabs(h1_err));
		l2_err = sqrt(fabs(l2_err));

		lp_err = pow(fabs(lp_err), 1. / p);

		// pred_norm = pow(fabs(pred_norm), 1./p);

		timer.stop();
		const double computing_errors_time = timer.getElapsedTime();
		logger().info(" took {}s", computing_errors_time);

		logger().info("-- L2 error: {}", l2_err);
		logger().info("-- Lp error: {}", lp_err);
		logger().info("-- H1 error: {}", h1_err);
		logger().info("-- H1 semi error: {}", h1_semi_err);
		// logger().info("-- Perd norm: {}", pred_norm);

		logger().info("-- Linf error: {}", linf_err);
		logger().info("-- grad max error: {}", grad_max_err);

		// {
		// 	std::ofstream out("errs.txt");
		// 	out<<err_per_el;
		// 	out.close();
		// }
	}

	void OutStatsData::compute_mesh_stats(const polyfem::mesh::Mesh &mesh)
	{
		using namespace polyfem::mesh;

		simplex_count = 0;
		regular_count = 0;
		regular_boundary_count = 0;
		simple_singular_count = 0;
		multi_singular_count = 0;
		boundary_count = 0;
		non_regular_boundary_count = 0;
		non_regular_count = 0;
		undefined_count = 0;
		multi_singular_boundary_count = 0;

		const auto &els_tag = mesh.elements_tag();

		for (size_t i = 0; i < els_tag.size(); ++i)
		{
			const ElementType type = els_tag[i];

			switch (type)
			{
			case ElementType::SIMPLEX:
				simplex_count++;
				break;
			case ElementType::REGULAR_INTERIOR_CUBE:
				regular_count++;
				break;
			case ElementType::REGULAR_BOUNDARY_CUBE:
				regular_boundary_count++;
				break;
			case ElementType::SIMPLE_SINGULAR_INTERIOR_CUBE:
				simple_singular_count++;
				break;
			case ElementType::MULTI_SINGULAR_INTERIOR_CUBE:
				multi_singular_count++;
				break;
			case ElementType::SIMPLE_SINGULAR_BOUNDARY_CUBE:
				boundary_count++;
				break;
			case ElementType::INTERFACE_CUBE:
			case ElementType::MULTI_SINGULAR_BOUNDARY_CUBE:
				multi_singular_boundary_count++;
				break;
			case ElementType::BOUNDARY_POLYTOPE:
				non_regular_boundary_count++;
				break;
			case ElementType::INTERIOR_POLYTOPE:
				non_regular_count++;
				break;
			case ElementType::UNDEFINED:
				undefined_count++;
				break;
			default:
				throw std::runtime_error("Unknown element type");
			}
		}

		logger().info("simplex_count: \t{}", simplex_count);
		logger().info("regular_count: \t{}", regular_count);
		logger().info("regular_boundary_count: \t{}", regular_boundary_count);
		logger().info("simple_singular_count: \t{}", simple_singular_count);
		logger().info("multi_singular_count: \t{}", multi_singular_count);
		logger().info("boundary_count: \t{}", boundary_count);
		logger().info("multi_singular_boundary_count: \t{}", multi_singular_boundary_count);
		logger().info("non_regular_count: \t{}", non_regular_count);
		logger().info("non_regular_boundary_count: \t{}", non_regular_boundary_count);
		logger().info("undefined_count: \t{}", undefined_count);
		logger().info("total count:\t {}", mesh.n_elements());
	}

	void OutStatsData::save_json(
		const nlohmann::json &args,
		const int n_bases, const int n_pressure_bases,
		const Eigen::MatrixXd &sol,
		const mesh::Mesh &mesh,
		const Eigen::VectorXi &disc_orders,
		const assembler::Problem &problem,
		const OutRuntimeData &runtime,
		const std::string &formulation,
		const bool isoparametric,
		const int sol_at_node_id,
		nlohmann::json &j)
	{

		j["args"] = args;

		j["geom_order"] = mesh.orders().size() > 0 ? mesh.orders().maxCoeff() : 1;
		j["geom_order_min"] = mesh.orders().size() > 0 ? mesh.orders().minCoeff() : 1;
		j["discr_order_min"] = disc_orders.minCoeff();
		j["discr_order_max"] = disc_orders.maxCoeff();
		j["iso_parametric"] = isoparametric;
		j["problem"] = problem.name();
		j["mat_size"] = mat_size;
		j["num_bases"] = n_bases;
		j["num_pressure_bases"] = n_pressure_bases;
		j["num_non_zero"] = nn_zero;
		j["num_flipped"] = n_flipped;
		j["num_dofs"] = num_dofs;
		j["num_vertices"] = mesh.n_vertices();
		j["num_elements"] = mesh.n_elements();

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

		j["time_building_basis"] = runtime.building_basis_time;
		j["time_loading_mesh"] = runtime.loading_mesh_time;
		j["time_computing_poly_basis"] = runtime.computing_poly_basis_time;
		j["time_assembling_stiffness_mat"] = runtime.assembling_stiffness_mat_time;
		j["time_assembling_mass_mat"] = runtime.assembling_mass_mat_time;
		j["time_assigning_rhs"] = runtime.assigning_rhs_time;
		j["time_solving"] = runtime.solving_time;
		// j["time_computing_errors"] = runtime.computing_errors_time;

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

		j["is_simplicial"] = mesh.n_elements() == simplex_count;

		j["peak_memory"] = getPeakRSS() / (1024 * 1024);

		const int actual_dim = problem.is_scalar() ? 1 : mesh.dimension();

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

		if (sol_at_node_id >= 0)
		{
			const int node_id = sol_at_node_id;

			for (int d = 0; d < actual_dim; ++d)
			{
				sol_at_node[d] = sol(node_id * actual_dim + d);
			}
		}

		j["sol_at_node"] = sol_at_node;
		j["sol_min"] = mmin;
		j["sol_max"] = mmax;

#if defined(POLYFEM_WITH_CPP_THREADS)
		j["num_threads"] = utils::get_n_threads();
#elif defined(POLYFEM_WITH_TBB)
		j["num_threads"] = utils::get_n_threads();
#else
		j["num_threads"] = 1;
#endif

		j["formulation"] = formulation;

		logger().info("done");
	}

	EnergyCSVWriter::EnergyCSVWriter(const std::string &path, const solver::SolveData &solve_data)
		: file(path), solve_data(solve_data)
	{
		file << "i,";
		for (const auto &[name, _] : solve_data.named_forms())
		{
			file << name << ",";
		}
		file << "total_energy" << std::endl;
	}

	EnergyCSVWriter::~EnergyCSVWriter()
	{
		file.close();
	}

	void EnergyCSVWriter::write(const int i, const Eigen::MatrixXd &sol)
	{
		const double s = solve_data.time_integrator
							 ? solve_data.time_integrator->acceleration_scaling()
							 : 1;
		file << i << ",";
		for (const auto &[_, form] : solve_data.named_forms())
		{
			// Divide by acceleration scaling to get the energy (units of J)
			file << ((form && form->enabled()) ? form->value(sol) : 0) / s << ",";
		}
		file << solve_data.nl_problem->value(sol) / s << "\n";
		file.flush();
	}

	RuntimeStatsCSVWriter::RuntimeStatsCSVWriter(const std::string &path, const State &state, const double t0, const double dt)
		: file(path), state(state), t0(t0), dt(dt)
	{
		file << "step,time,forward,remeshing,global_relaxation,peak_mem,#V,#T" << std::endl;
	}

	RuntimeStatsCSVWriter::~RuntimeStatsCSVWriter()
	{
		file.close();
	}

	void RuntimeStatsCSVWriter::write(const int t, const double forward, const double remeshing, const double global_relaxation, const Eigen::MatrixXd &sol)
	{
		total_forward_solve_time += forward;
		total_remeshing_time += remeshing;
		total_global_relaxation_time += global_relaxation;

		// logger().debug(
		// 	"Forward (cur, avg, total): {} s, {} s, {} s",
		// 	forward, total_forward_solve_time / t, total_forward_solve_time);
		// logger().debug(
		// 	"Remeshing (cur, avg, total): {} s, {} s, {} s",
		// 	remeshing, total_remeshing_time / t, total_remeshing_time);
		// logger().debug(
		// 	"Global relaxation (cur, avg, total): {} s, {} s, {} s",
		// 	global_relaxation, total_global_relaxation_time / t, total_global_relaxation_time);

		const double peak_mem = getPeakRSS() / double(1 << 30);
		// logger().debug("Peak mem: {} GiB", peak_mem);

		file << fmt::format(
			"{},{},{},{},{},{},{},{}\n",
			t, t0 + dt * t, forward, remeshing, global_relaxation, peak_mem,
			state.n_bases, state.mesh->n_elements());
		file.flush();
	}

} // namespace polyfem::io

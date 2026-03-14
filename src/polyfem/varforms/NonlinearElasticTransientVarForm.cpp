#include "NonlinearElasticTransientVarForm.hpp"

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/assembler/MultiModel.hpp>

#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/mesh/collision_proxy/CollisionProxy.hpp>

#include <polyfem/refinement/APriori.hpp>

#include <polyfem/basis/LagrangeBasis2d.hpp>
#include <polyfem/basis/LagrangeBasis3d.hpp>
#include <polyfem/basis/SplineBasis2d.hpp>
#include <polyfem/basis/SplineBasis3d.hpp>
#include <polyfem/basis/barycentric/MVPolygonalBasis2d.hpp>
#include <polyfem/basis/barycentric/WSPolygonalBasis2d.hpp>
#include <polyfem/basis/PolygonalBasis2d.hpp>
#include <polyfem/basis/PolygonalBasis3d.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Timer.hpp>

#include <polyfem/problem/KernelProblem.hpp>
#include <polyfem/problem/ProblemFactory.hpp>

#include <igl/Timer.h>
#include <igl/edges.h>

namespace polyfem::varform
{
	namespace
	{
		/// Assumes in nodes are in order vertex, edge, face, then cell nodes.
		void build_in_node_to_in_primitive(const mesh::Mesh &mesh, const mesh::MeshNodes &mesh_nodes,
										   Eigen::VectorXi &in_node_to_in_primitive,
										   Eigen::VectorXi &in_node_offset)
		{
			const int num_vertex_nodes = mesh_nodes.num_vertex_nodes();
			const int num_edge_nodes = mesh_nodes.num_edge_nodes();
			const int num_face_nodes = mesh_nodes.num_face_nodes();
			const int num_cell_nodes = mesh_nodes.num_cell_nodes();

			const int num_nodes = num_vertex_nodes + num_edge_nodes + num_face_nodes + num_cell_nodes;

			const long n_vertices = num_vertex_nodes;
			const int num_in_primitives = n_vertices + mesh.n_edges() + mesh.n_faces() + mesh.n_cells();
			const int num_primitives = mesh.n_vertices() + mesh.n_edges() + mesh.n_faces() + mesh.n_cells();

			in_node_to_in_primitive.resize(num_nodes);
			in_node_offset.resize(num_nodes);

			// Only one node per vertex, so this is an identity map.
			in_node_to_in_primitive.head(num_vertex_nodes).setLinSpaced(num_vertex_nodes, 0, num_vertex_nodes - 1); // vertex nodes
			in_node_offset.head(num_vertex_nodes).setZero();

			int prim_offset = n_vertices;
			int node_offset = num_vertex_nodes;
			auto foo = [&](const int num_prims, const int num_prim_nodes) {
				if (num_prims <= 0 || num_prim_nodes <= 0)
					return;
				const Eigen::VectorXi range = Eigen::VectorXi::LinSpaced(num_prim_nodes, 0, num_prim_nodes - 1);
				// TODO: This assumes isotropic degree of element.
				const int node_per_prim = num_prim_nodes / num_prims;

				in_node_to_in_primitive.segment(node_offset, num_prim_nodes) =
					range.array() / node_per_prim + prim_offset;

				in_node_offset.segment(node_offset, num_prim_nodes) =
					range.unaryExpr([&](const int x) { return x % node_per_prim; });

				prim_offset += num_prims;
				node_offset += num_prim_nodes;
			};

			foo(mesh.n_edges(), num_edge_nodes);
			foo(mesh.n_faces(), num_face_nodes);
			foo(mesh.n_cells(), num_cell_nodes);
		}

		bool build_in_primitive_to_primitive(
			const mesh::Mesh &mesh, const mesh::MeshNodes &mesh_nodes,
			const Eigen::VectorXi &in_ordered_vertices,
			const Eigen::MatrixXi &in_ordered_edges,
			const Eigen::MatrixXi &in_ordered_faces,
			Eigen::VectorXi &in_primitive_to_primitive)
		{
			// NOTE: Assume in_cells_to_cells is identity
			const int num_vertex_nodes = mesh_nodes.num_vertex_nodes();
			const int num_edge_nodes = mesh_nodes.num_edge_nodes();
			const int num_face_nodes = mesh_nodes.num_face_nodes();
			const int num_cell_nodes = mesh_nodes.num_cell_nodes();
			const int num_nodes = num_vertex_nodes + num_edge_nodes + num_face_nodes + num_cell_nodes;

			const long n_vertices = num_vertex_nodes;
			const int num_in_primitives = n_vertices + mesh.n_edges() + mesh.n_faces() + mesh.n_cells();
			const int num_primitives = mesh.n_vertices() + mesh.n_edges() + mesh.n_faces() + mesh.n_cells();

			in_primitive_to_primitive.setLinSpaced(num_in_primitives, 0, num_in_primitives - 1);

			igl::Timer timer;

			// ------------
			// Map vertices
			// ------------

			if (in_ordered_vertices.rows() != n_vertices)
			{
				logger().warn("Node ordering disabled, in_ordered_vertices != n_vertices, {} != {}", in_ordered_vertices.rows(), n_vertices);
				return false;
			}

			in_primitive_to_primitive.head(n_vertices) = in_ordered_vertices;

			int in_offset = n_vertices;
			int offset = mesh.n_vertices();

			// ---------
			// Map edges
			// ---------

			logger().trace("Building Mesh edges to IDs...");
			timer.start();
			const auto edges_to_ids = mesh.edges_to_ids();
			if (in_ordered_edges.rows() != edges_to_ids.size())
			{
				logger().warn("Node ordering disabled, in_ordered_edges != edges_to_ids, {} != {}", in_ordered_edges.rows(), edges_to_ids.size());
				return false;
			}
			timer.stop();
			logger().trace("Done (took {}s)", timer.getElapsedTime());

			logger().trace("Building in-edge to edge mapping...");
			timer.start();
			for (int in_ei = 0; in_ei < in_ordered_edges.rows(); in_ei++)
			{
				const std::pair<int, int> in_edge(
					in_ordered_edges.row(in_ei).minCoeff(),
					in_ordered_edges.row(in_ei).maxCoeff());
				in_primitive_to_primitive[in_offset + in_ei] =
					offset + edges_to_ids.at(in_edge); // offset edge ids
			}
			timer.stop();
			logger().trace("Done (took {}s)", timer.getElapsedTime());

			in_offset += mesh.n_edges();
			offset += mesh.n_edges();

			// ---------
			// Map faces
			// ---------

			if (mesh.is_volume())
			{
				logger().trace("Building Mesh faces to IDs...");
				timer.start();
				const auto faces_to_ids = mesh.faces_to_ids();
				if (in_ordered_faces.rows() != faces_to_ids.size())
				{
					logger().warn("Node ordering disabled, in_ordered_faces != faces_to_ids, {} != {}", in_ordered_faces.rows(), faces_to_ids.size());
					return false;
				}
				timer.stop();
				logger().trace("Done (took {}s)", timer.getElapsedTime());

				logger().trace("Building in-face to face mapping...");
				timer.start();
				for (int in_fi = 0; in_fi < in_ordered_faces.rows(); in_fi++)
				{
					std::vector<int> in_face(in_ordered_faces.cols());
					for (int i = 0; i < in_face.size(); i++)
						in_face[i] = in_ordered_faces(in_fi, i);
					std::sort(in_face.begin(), in_face.end());

					in_primitive_to_primitive[in_offset + in_fi] =
						offset + faces_to_ids.at(in_face); // offset face ids
				}
				timer.stop();
				logger().trace("Done (took {}s)", timer.getElapsedTime());

				in_offset += mesh.n_faces();
				offset += mesh.n_faces();
			}

			return true;
		}
	} // namespace

	void NonlinearElasticTransientVarForm::reset()
	{
		VarForm::reset();

		bases.clear();
		geom_bases_.clear();
		boundary_nodes.clear();
		dirichlet_nodes.clear();
		neumann_nodes.clear();
		local_boundary.clear();
		total_local_boundary.clear();
		local_neumann_boundary.clear();
		local_pressure_boundary.clear();
		local_pressure_cavity.clear();
		polys.clear();
		poly_edge_to_data.clear();

		n_bases = 0;
		n_geom_bases = 0;
	}

	void NonlinearElasticTransientVarForm::init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
	{
		VarForm::init(formulation, units, args, out_path);

		assembler = assembler::AssemblerUtils::make_assembler(formulation);
		assert(assembler->name() == formulation);
		mass_matrix_assembler = std::make_shared<assembler::Mass>();

		if (args["solver"]["advanced"]["check_inversion"] == "Conservative")
		{
			if (auto elastic_assembler = std::dynamic_pointer_cast<assembler::ElasticityAssembler>(assembler))
				elastic_assembler->set_use_robust_jacobian();
		}

		if (!args.contains("preset_problem"))
		{
			problem = std::make_shared<assembler::GenericTensorProblem>("GenericTensor");

			problem->clear();
			const auto tmp = R"({"is_time_dependent": true})"_json;
			problem->set_parameters(tmp);

			// important for the BC

			auto bc = args["boundary_conditions"];
			bc["root_path"] = root_path;
			problem->set_parameters(bc);
			problem->set_parameters(args["initial_conditions"]);
			problem->set_parameters(args["output"]);
		}
		else
		{
			if (args["preset_problem"]["type"] == "Kernel")
			{
				problem = std::make_shared<problem::KernelProblem>("Kernel", *assembler);
				problem->clear();
				problem::KernelProblem &kprob = *dynamic_cast<problem::KernelProblem *>(problem.get());
			}
			else
			{
				problem = problem::ProblemFactory::factory().get_problem(args["preset_problem"]["type"]);
				problem->clear();
			}
			// important for the BC
			problem->set_parameters(args["preset_problem"]);
		}

		problem->set_units(*assembler, units);

		t0 = args["time"]["t0"];
		time_steps = args["time"]["time_steps"];
		dt = args["time"]["dt"];
	}

	void NonlinearElasticTransientVarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		if (assembler::MultiModel *mm = dynamic_cast<assembler::MultiModel *>(assembler.get()))
		{
			assert(args["materials"].is_array());

			std::vector<std::string> materials(mesh.n_elements());

			std::map<int, std::string> mats;

			for (const auto &m : args["materials"])
				mats[m["id"].get<int>()] = m["type"];

			for (int i = 0; i < materials.size(); ++i)
				materials[i] = mats.at(mesh.get_body_id(i));

			mm->init_multimodels(materials);
		}

		problem->init(mesh);
	}

	void NonlinearElasticTransientVarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
	{
		using namespace mesh;
		this->iso_parametric = iso_parametric;

		VarForm::assign_discr_orders(args["space"]["discr_order"], mesh, disc_orders);

		Eigen::MatrixXi geom_disc_orders;
		if (!iso_parametric)
		{
			if (mesh.orders().size() <= 0)
			{
				geom_disc_orders.resizeLike(disc_orders);
				geom_disc_orders.setConstant(1);
			}
			else
				geom_disc_orders = mesh.orders();
		}

		Eigen::MatrixXi geom_disc_ordersq = geom_disc_orders;
		disc_ordersq = disc_orders;

		igl::Timer timer;
		timer.start();
		if (args["space"]["use_p_ref"])
		{
			refinement::APriori::p_refine(
				mesh,
				args["space"]["advanced"]["B"],
				args["space"]["advanced"]["h1_formula"],
				args["space"]["discr_order"],
				args["space"]["advanced"]["discr_order_max"],
				stats,
				disc_orders);

			logger().info("min p: {} max p: {}", disc_orders.minCoeff(), disc_orders.maxCoeff());
		}

		logger().info("Building {} basis...", (iso_parametric ? "isoparametric" : "not isoparametric"));
		const bool has_polys = mesh.has_poly();

		local_boundary.clear();
		local_neumann_boundary.clear();
		local_pressure_boundary.clear();
		local_pressure_cavity.clear();
		std::map<int, basis::InterfaceData> poly_edge_to_data_geom; // temp dummy variable

		int quadrature_order = args["space"]["advanced"]["quadrature_order"].get<int>();
		const int mass_quadrature_order = args["space"]["advanced"]["mass_quadrature_order"].get<int>();

		// shape optimization needs continuous geometric basis
		// const bool use_continuous_gbasis = optimization_enabled == solver::CacheLevel::Derivatives;
		const bool use_continuous_gbasis = true;
		const bool use_corner_quadrature = args["space"]["advanced"]["use_corner_quadrature"];

		if (mesh.is_volume())
		{
			const Mesh3D &tmp_mesh = dynamic_cast<const Mesh3D &>(mesh);
			if (args["space"]["basis_type"] == "Spline")
			{
				n_bases = basis::SplineBasis3d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, bases, local_boundary, poly_edge_to_data);
			}
			else
			{
				if (!iso_parametric)
					n_geom_bases = basis::LagrangeBasis3d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, geom_disc_orders, geom_disc_ordersq, false, false, has_polys, !use_continuous_gbasis, use_corner_quadrature, geom_bases_, local_boundary, poly_edge_to_data_geom, geom_mesh_nodes);

				n_bases = basis::LagrangeBasis3d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, disc_orders, disc_ordersq, args["space"]["basis_type"] == "Bernstein", args["space"]["basis_type"] == "Serendipity", has_polys, false, use_corner_quadrature, bases, local_boundary, poly_edge_to_data, mesh_nodes);
			}
		}
		else
		{
			const Mesh2D &tmp_mesh = dynamic_cast<const Mesh2D &>(mesh);
			if (args["space"]["basis_type"] == "Spline")
			{
				// TODO:
				// if (!iso_parametric())
				// {
				// 	logger().error("Splines must be isoparametric, ignoring...");
				// 	// LagrangeBasis2d::build_bases(tmp_mesh, quadrature_order, disc_orders, has_polys, geom_bases_, local_boundary, poly_edge_to_data_geom, mesh_nodes);
				// 	n_bases = SplineBasis2d::build_bases(tmp_mesh, quadrature_order, geom_bases_, local_boundary, poly_edge_to_data);
				// }

				n_bases = basis::SplineBasis2d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, bases, local_boundary, poly_edge_to_data);

				// if (iso_parametric() && args["fit_nodes"])
				// 	SplineBasis2d::fit_nodes(tmp_mesh, n_bases, bases);
			}
			else
			{
				if (!iso_parametric)
					n_geom_bases = basis::LagrangeBasis2d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, geom_disc_orders, false, false, has_polys, !use_continuous_gbasis, use_corner_quadrature, geom_bases_, local_boundary, poly_edge_to_data_geom, geom_mesh_nodes);

				n_bases = basis::LagrangeBasis2d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, disc_orders, args["space"]["basis_type"] == "Bernstein", args["space"]["basis_type"] == "Serendipity", has_polys, false, use_corner_quadrature, bases, local_boundary, poly_edge_to_data, mesh_nodes);
			}
		}

		timer.stop();

		// FIXME!!
		// build_polygonal_basis();

		if (n_geom_bases == 0)
			n_geom_bases = n_bases;

		for (const auto &lb : local_boundary)
			total_local_boundary.emplace_back(lb);

		const int dim = mesh.dimension();
		// FIXME!! handle periodic bc
		//  handle periodic bc
		//  if (has_periodic_bc())
		//  {
		//  	// collect periodic directions
		//  	json directions = args["boundary_conditions"]["periodic_boundary"]["correspondence"];
		//  	Eigen::MatrixXd tile_offset = Eigen::MatrixXd::Identity(dim, dim);

		// 	if (directions.size() > 0)
		// 	{
		// 		Eigen::VectorXd tmp;
		// 		for (int d = 0; d < dim; d++)
		// 		{
		// 			tmp = directions[d];
		// 			if (tmp.size() != dim)
		// 				log_and_throw_error("Invalid size of periodic directions!");
		// 			tile_offset.col(d) = tmp;
		// 		}
		// 	}

		// 	periodic_bc = std::make_shared<PeriodicBoundary>(problem->is_scalar(), n_bases, bases, mesh_nodes, tile_offset, args["boundary_conditions"]["periodic_boundary"]["tolerance"].get<double>());

		// 	macro_strain_constraint.init(dim, args["boundary_conditions"]["periodic_boundary"]);
		// }

		if (args["space"]["advanced"]["count_flipped_els"])
			stats.count_flipped_elements(mesh, geom_bases());

		const int prev_bases = n_bases;
		n_bases += obstacle.n_vertices();

		{
			igl::Timer timer2;
			logger().debug("Building node mapping...");
			timer2.start();
			build_node_mapping(mesh, args);
			problem->update_nodes(in_node_to_node);
			mesh.update_nodes(in_node_to_node);
			timer2.stop();
			logger().debug("Done (took {}s)", timer2.getElapsedTime());
		}

		logger().info("Building collision mesh...");
		build_collision_mesh(mesh, args);
		// FIXME!! handle periodic collision mesh
		//  if (periodic_bc && args["contact"]["periodic"])
		//  	build_periodic_collision_mesh();
		logger().info("Done!");

		// FIXME remove pressure
		std::vector<int> tmp;
		const int prev_b_size = local_boundary.size();
		problem->setup_bc(mesh, n_bases - obstacle.n_vertices(),
						  bases, geom_bases(), {},
						  local_boundary,
						  boundary_nodes,
						  local_neumann_boundary,
						  local_pressure_boundary,
						  local_pressure_cavity,
						  tmp,
						  dirichlet_nodes, neumann_nodes);

		// setp nodal values
		{
			dirichlet_nodes_position.resize(dirichlet_nodes.size());
			for (int n = 0; n < dirichlet_nodes.size(); ++n)
			{
				const int n_id = dirichlet_nodes[n];
				bool found = false;
				for (const auto &bs : bases)
				{
					for (const auto &b : bs.bases)
					{
						for (const auto &lg : b.global())
						{
							if (lg.index == n_id)
							{
								dirichlet_nodes_position[n] = lg.node;
								found = true;
								break;
							}
						}

						if (found)
							break;
					}

					if (found)
						break;
				}

				assert(found);
			}

			neumann_nodes_position.resize(neumann_nodes.size());
			for (int n = 0; n < neumann_nodes.size(); ++n)
			{
				const int n_id = neumann_nodes[n];
				bool found = false;
				for (const auto &bs : bases)
				{
					for (const auto &b : bs.bases)
					{
						for (const auto &lg : b.global())
						{
							if (lg.index == n_id)
							{
								neumann_nodes_position[n] = lg.node;
								found = true;
								break;
							}
						}

						if (found)
							break;
					}

					if (found)
						break;
				}

				assert(found);
			}
		}

		for (int i = prev_bases; i < n_bases; ++i)
		{
			for (int d = 0; d < dim; ++d)
				boundary_nodes.push_back(i * dim + d);
		}

		std::sort(boundary_nodes.begin(), boundary_nodes.end());
		auto it = std::unique(boundary_nodes.begin(), boundary_nodes.end());
		boundary_nodes.resize(std::distance(boundary_nodes.begin(), it));

		const auto &curret_bases = geom_bases();
		const int n_samples = 10;
		stats.compute_mesh_size(mesh, curret_bases, n_samples, args["output"]["advanced"]["curved_mesh_size"]);

		// FIXME fix me
		//  if (is_contact_enabled())
		//  {
		//  	double min_boundary_edge_length = std::numeric_limits<double>::max();
		//  	for (const auto &edge : collision_mesh.edges().rowwise())
		//  	{
		//  		const VectorNd v0 = collision_mesh.rest_positions().row(edge(0));
		//  		const VectorNd v1 = collision_mesh.rest_positions().row(edge(1));
		//  		min_boundary_edge_length = std::min(min_boundary_edge_length, (v1 - v0).norm());
		//  	}

		// 	double dhat = Units::convert(args["contact"]["dhat"], units.length());
		// 	args["contact"]["epsv"] = Units::convert(args["contact"]["epsv"], units.velocity());
		// 	args["contact"]["dhat"] = dhat;

		// 	if (!has_dhat && dhat > min_boundary_edge_length)
		// 	{
		// 		args["contact"]["dhat"] = double(args["contact"]["dhat_percentage"]) * min_boundary_edge_length;
		// 		logger().info("dhat set to {}", double(args["contact"]["dhat"]));
		// 	}
		// 	else
		// 	{
		// 		if (dhat > min_boundary_edge_length)
		// 			logger().warn("dhat larger than min boundary edge, {} > {}", dhat, min_boundary_edge_length);
		// 	}
		// }

		logger().info("n_bases {}", n_bases);

		timings.building_basis_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.building_basis_time);

		logger().info("flipped elements {}", stats.n_flipped);
		logger().info("h: {}", stats.mesh_size);
		logger().info("n bases: {}", n_bases);

		if (n_bases <= args["solver"]["advanced"]["cache_size"])
		{
			timer.start();
			logger().info("Building cache...");
			ass_vals_cache.init(mesh.is_volume(), bases, curret_bases);
			mass_ass_vals_cache.init(mesh.is_volume(), bases, curret_bases, true);

			logger().info(" took {}s", timer.getElapsedTime());
		}
		else
		{
			ass_vals_cache.init_empty();
			mass_ass_vals_cache.init_empty(true);
		}

		// FIXME
		//  out_geom.build_grid(*mesh, args["output"]["advanced"]["sol_on_grid"]);

		{
			json rhs_solver_params = args["solver"]["linear"];
			if (!rhs_solver_params.contains("Pardiso"))
				rhs_solver_params["Pardiso"] = {};
			rhs_solver_params["Pardiso"]["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

			solve_data.rhs_assembler = std::make_shared<assembler::RhsAssembler>(
				*assembler, mesh, obstacle,
				dirichlet_nodes, neumann_nodes,
				dirichlet_nodes_position, neumann_nodes_position,
				n_bases, dim, bases, geom_bases(), ass_vals_cache, *problem,
				args["space"]["advanced"]["bc_method"],
				rhs_solver_params);
		}
	}

	void NonlinearElasticTransientVarForm::build_node_mapping(const mesh::Mesh &mesh, const json &args)
	{
		if (args["space"]["basis_type"] == "Spline")
		{
			logger().warn("Node ordering disabled, it dosent work for splines!");
			return;
		}

		if (disc_orders.maxCoeff() >= 4 || disc_orders.maxCoeff() != disc_orders.minCoeff())
		{
			logger().warn("Node ordering disabled, it works only for p < 4 and uniform order!");
			return;
		}

		if (!mesh.is_conforming())
		{
			logger().warn("Node ordering disabled, not supported for non-conforming meshes!");
			return;
		}

		if (mesh.has_poly())
		{
			logger().warn("Node ordering disabled, not supported for polygonal meshes!");
			return;
		}

		if (mesh.in_ordered_vertices().size() <= 0 || mesh.in_ordered_edges().size() <= 0 || (mesh.is_volume() && mesh.in_ordered_faces().size() <= 0))
		{
			logger().warn("Node ordering disabled, input vertices/edges/faces not computed!");
			return;
		}

		const int num_vertex_nodes = mesh_nodes->num_vertex_nodes();
		const int num_edge_nodes = mesh_nodes->num_edge_nodes();
		const int num_face_nodes = mesh_nodes->num_face_nodes();
		const int num_cell_nodes = mesh_nodes->num_cell_nodes();

		const int num_nodes = num_vertex_nodes + num_edge_nodes + num_face_nodes + num_cell_nodes;

		const long n_vertices = num_vertex_nodes;
		const int num_in_primitives = n_vertices + mesh.n_edges() + mesh.n_faces() + mesh.n_cells();
		const int num_primitives = mesh.n_vertices() + mesh.n_edges() + mesh.n_faces() + mesh.n_cells();

		igl::Timer timer;

		logger().trace("Building in-node to in-primitive mapping...");
		timer.start();
		Eigen::VectorXi in_node_to_in_primitive;
		Eigen::VectorXi in_node_offset;
		build_in_node_to_in_primitive(mesh, *mesh_nodes, in_node_to_in_primitive, in_node_offset);
		timer.stop();
		logger().trace("Done (took {}s)", timer.getElapsedTime());

		logger().trace("Building in-primitive to primitive mapping...");
		timer.start();
		bool ok = build_in_primitive_to_primitive(
			mesh, *mesh_nodes,
			mesh.in_ordered_vertices(),
			mesh.in_ordered_edges(),
			mesh.in_ordered_faces(),
			in_primitive_to_primitive);
		timer.stop();
		logger().trace("Done (took {}s)", timer.getElapsedTime());

		if (!ok)
		{
			in_node_to_node.resize(0);
			in_primitive_to_primitive.resize(0);
			return;
		}
		const auto &tmp = mesh_nodes->in_ordered_vertices();
		int max_tmp = -1;
		for (auto v : tmp)
			max_tmp = std::max(max_tmp, v);

		in_node_to_node.resize(max_tmp + 1);
		for (int i = 0; i < tmp.size(); ++i)
		{
			if (tmp[i] >= 0)
				in_node_to_node[tmp[i]] = i;
		}
	}

	void NonlinearElasticTransientVarForm::build_collision_mesh(
		const mesh::Mesh &mesh,
		const json &args)
	{
		build_collision_mesh(
			mesh, n_bases, bases, geom_bases(), total_local_boundary, obstacle,
			args, [this](const std::string &p) { return utils::resolve_path(p, root_path, false); },
			in_node_to_node, collision_mesh);
	}

	void NonlinearElasticTransientVarForm::build_collision_mesh(
		const mesh::Mesh &mesh,
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		const mesh::Obstacle &obstacle,
		const json &args,
		const std::function<std::string(const std::string &)> &resolve_input_path,
		const Eigen::VectorXi &in_node_to_node,
		ipc::CollisionMesh &collision_mesh)
	{
		Eigen::MatrixXd collision_vertices;
		Eigen::VectorXi collision_codim_vids;
		Eigen::MatrixXi collision_edges, collision_triangles;
		std::vector<Eigen::Triplet<double>> displacement_map_entries;

		if (args.contains("/contact/collision_mesh"_json_pointer)
			&& args.at("/contact/collision_mesh/enabled"_json_pointer).get<bool>())
		{
			const json collision_mesh_args = args.at("/contact/collision_mesh"_json_pointer);
			if (collision_mesh_args.contains("linear_map"))
			{
				assert(displacement_map_entries.empty());
				assert(collision_mesh_args.contains("mesh"));
				const std::string root_path = utils::json_value<std::string>(args, "root_path", "");
				// TODO: handle transformation per geometry
				const json transformation = utils::json_as_array(args["geometry"])[0]["transformation"];
				mesh::load_collision_proxy(
					utils::resolve_path(collision_mesh_args["mesh"], root_path),
					utils::resolve_path(collision_mesh_args["linear_map"], root_path),
					in_node_to_node, transformation, collision_vertices, collision_codim_vids,
					collision_edges, collision_triangles, displacement_map_entries);
			}
			else if (collision_mesh_args.contains("max_edge_length"))
			{
				logger().debug(
					"Building collision proxy with max edge length={} ...",
					collision_mesh_args["max_edge_length"].get<double>());
				igl::Timer timer;
				timer.start();
				build_collision_proxy(
					bases, geom_bases, total_local_boundary, n_bases, mesh.dimension(),
					collision_mesh_args["max_edge_length"], collision_vertices,
					collision_triangles, displacement_map_entries,
					collision_mesh_args["tessellation_type"]);
				if (collision_triangles.size())
					igl::edges(collision_triangles, collision_edges);
				timer.stop();
				logger().debug(fmt::format(
					std::locale("en_US.UTF-8"),
					"Done (took {:g}s, {:L} vertices, {:L} triangles)",
					timer.getElapsedTime(),
					collision_vertices.rows(), collision_triangles.rows()));
			}
			else
			{
				io::OutGeometryData::extract_boundary_mesh(
					mesh, n_bases - obstacle.n_vertices(), bases, total_local_boundary,
					collision_vertices, collision_edges, collision_triangles, displacement_map_entries);
			}
		}
		else
		{
			io::OutGeometryData::extract_boundary_mesh(
				mesh, n_bases - obstacle.n_vertices(), bases, total_local_boundary,
				collision_vertices, collision_edges, collision_triangles, displacement_map_entries);
		}

		std::vector<bool> is_orientable_vertex(collision_vertices.rows(), true);

		// n_bases already contains the obstacle vertices
		const int num_fe_nodes = n_bases - obstacle.n_vertices();
		const int num_fe_collision_vertices = collision_vertices.rows();
		assert(collision_edges.size() == 0 || collision_edges.maxCoeff() < num_fe_collision_vertices);
		assert(collision_triangles.size() == 0 || collision_triangles.maxCoeff() < num_fe_collision_vertices);

		// Append the obstacles to the collision mesh
		if (obstacle.n_vertices() > 0)
		{
			utils::append_rows(collision_vertices, obstacle.v());
			utils::append_rows(collision_codim_vids, obstacle.codim_v().array() + num_fe_collision_vertices);
			utils::append_rows(collision_edges, obstacle.e().array() + num_fe_collision_vertices);
			utils::append_rows(collision_triangles, obstacle.f().array() + num_fe_collision_vertices);

			for (int i = 0; i < obstacle.n_vertices(); i++)
			{
				is_orientable_vertex.push_back(false);
			}

			if (!displacement_map_entries.empty())
			{
				displacement_map_entries.reserve(displacement_map_entries.size() + obstacle.n_vertices());
				for (int i = 0; i < obstacle.n_vertices(); i++)
				{
					displacement_map_entries.emplace_back(num_fe_collision_vertices + i, num_fe_nodes + i, 1.0);
				}
			}
		}

		std::vector<bool> is_on_surface = ipc::CollisionMesh::construct_is_on_surface(
			collision_vertices.rows(), collision_edges);
		for (const int vid : collision_codim_vids)
		{
			is_on_surface[vid] = true;
		}

		Eigen::SparseMatrix<double> displacement_map;
		if (!displacement_map_entries.empty())
		{
			displacement_map.resize(collision_vertices.rows(), n_bases);
			displacement_map.setFromTriplets(displacement_map_entries.begin(), displacement_map_entries.end());
		}

		collision_mesh = ipc::CollisionMesh(
			is_on_surface, is_orientable_vertex, collision_vertices, collision_edges, collision_triangles,
			displacement_map);

		collision_mesh.can_collide = [&collision_mesh, num_fe_collision_vertices](size_t vi, size_t vj) {
			// obstacles do not collide with other obstacles
			return collision_mesh.to_full_vertex_id(vi) < num_fe_collision_vertices
				   || collision_mesh.to_full_vertex_id(vj) < num_fe_collision_vertices;
		};

		collision_mesh.init_area_jacobians();
	}

	void NonlinearElasticTransientVarForm::solve(Eigen::MatrixXd &sol)
	{
		const bool save = true;
		const bool save_stats = false;
		const std::string filename = "";
		stats.spectrum.setZero();

		igl::Timer timer;
		timer.start();
		logger().info("Solving {}", assembler->name());

		{
			POLYFEM_SCOPED_TIMER("Setup RHS");

			// FIXME
			//  read_initial_x_from_file(
			//  resolve_input_path(args["input"]["data"]["state"]), "u",
			//  args["input"]["data"]["reorder"], in_node_to_node,
			//  mesh->dimension(), solution);

			if (sol.size() <= 0)
				solve_data.rhs_assembler->initial_solution(sol);

			if (sol.cols() > 1) // ignore previous solutions
				sol.conservativeResize(Eigen::NoChange, 1);
		}
		// Pre log the output path for easier watching
		if (save)
		{
			logger().info("Time sequence of simulation will be written to: \"{}\"",
						  resolve_output_path(filename));
		}

		init_solve(sol, t0 + dt);

		// Write the total energy to a CSV file
		int save_i = 0;

		std::unique_ptr<io::EnergyCSVWriter> energy_csv = nullptr;
		std::unique_ptr<io::RuntimeStatsCSVWriter> stats_csv = nullptr;

		if (save_stats)
		{
			logger().debug("Saving nl stats to {} and {}", resolve_output_path("energy.csv"), resolve_output_path("stats.csv"));
			energy_csv = std::make_unique<io::EnergyCSVWriter>(resolve_output_path("energy.csv"), solve_data);
			stats_csv = std::make_unique<io::RuntimeStatsCSVWriter>(resolve_output_path("stats.csv"), *this, t0, dt);
		}

		// Save the initial solution
		if (energy_csv)
			energy_csv->write(save_i, sol);

		save_timestep(t0, save_i, t0, dt, sol, Eigen::MatrixXd()); // no pressure
		save_i++;

		// Step 0.
		if (user_post_step)
		{
			user_post_step(0, *this, sol, nullptr, nullptr);
		}

		for (int t = 1; t <= time_steps; ++t)
		{
			double forward_solve_time = 0, remeshing_time = 0, global_relaxation_time = 0;

			{
				POLYFEM_SCOPED_TIMER(forward_solve_time);
				solve_tensor_nonlinear(t, sol, true);
			}

			if (remesh_enabled)
			{
				if (energy_csv)
					energy_csv->write(save_i, sol);
				// save_timestep(t0 + dt * t, save_i, t0, save_dt, sol, Eigen::MatrixXd()); // no pressure
				save_i++;

				bool remesh_success;
				{
					POLYFEM_SCOPED_TIMER(remeshing_time);
					remesh_success = this->remesh(t0 + dt * t, dt, sol);
				}

				// Save the solution after remeshing
				if (energy_csv)
					energy_csv->write(save_i, sol);
				// save_timestep(t0 + dt * t, save_i, t0, save_dt, sol, Eigen::MatrixXd()); // no pressure
				save_i++;

				// Only do global relaxation if remeshing was successful
				if (remesh_success)
				{
					POLYFEM_SCOPED_TIMER(global_relaxation_time);
					solve_tensor_nonlinear(t, sol, false); // solve the scene again after remeshing
				}
			}

			// Always save the solution for consistency
			if (energy_csv)
				energy_csv->write(save_i, sol);

			save_timestep(t0 + dt * t, t, t0, dt, sol, Eigen::MatrixXd()); // no pressure
			save_i++;

			if (user_post_step)
			{
				user_post_step(t, *this, sol, nullptr, nullptr);
			}

			{
				POLYFEM_SCOPED_TIMER("Update quantities");

				solve_data.time_integrator->update_quantities(sol);

				solve_data.nl_problem->update_quantities(t0 + (t + 1) * dt, sol);

				solve_data.update_dt();
				solve_data.update_barrier_stiffness(sol);
			}

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);
			if (time_callback)
				time_callback(t, time_steps, t0 + dt * t, t0 + dt * time_steps);

			const std::string rest_mesh_path = args["output"]["data"]["rest_mesh"].get<std::string>();
			if (!rest_mesh_path.empty())
			{
				Eigen::MatrixXd V;
				Eigen::MatrixXi F;
				build_mesh_matrices(V, F);
				io::MshWriter::write(
					resolve_output_path(fmt::format(args["output"]["data"]["rest_mesh"], t)),
					V, F, mesh->get_body_ids(), mesh->is_volume(), /*binary=*/true);
			}

			const std::string &state_path = resolve_output_path(fmt::format(args["output"]["data"]["state"], t));
			if (!state_path.empty())
				solve_data.time_integrator->save_state(state_path);

			// save restart file
			save_restart_json(t0, dt, t);
			if (remesh_enabled && stats_csv)
				stats_csv->write(t, forward_solve_time, remeshing_time, global_relaxation_time, sol);
		}

		timer.stop();
		timings.solving_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.solving_time);
	}

	void NonlinearElasticTransientVarForm::init_forms(const json &args, const int dim, Eigen::MatrixXd &sol, const double t)
	{
		damping_assembler = std::make_shared<assembler::ViscousDamping>();
		set_materials(*damping_assembler);

		elasticity_pressure_assembler = build_pressure_assembler();

		// for backward solve
		damping_prev_assembler = std::make_shared<assembler::ViscousDampingPrev>();
		set_materials(*damping_prev_assembler);

		const ElementInversionCheck check_inversion = args["solver"]["advanced"]["check_inversion"];

		forms = solve_data.init_forms(
			// General
			units,
			dim, t, in_node_to_node,
			// Elastic form
			n_bases, bases, geom_bases(), *assembler, ass_vals_cache, mass_ass_vals_cache, args["solver"]["advanced"]["jacobian_threshold"], check_inversion,
			// Body form
			n_pressure_bases, boundary_nodes, local_boundary,
			local_neumann_boundary,
			n_boundary_samples(), rhs, sol, mass_matrix_assembler->density(),
			// Pressure form
			local_pressure_boundary, local_pressure_cavity, elasticity_pressure_assembler,
			// Inertia form
			args.value("/time/quasistatic"_json_pointer, true), mass,
			damping_assembler->is_valid() ? damping_assembler : nullptr,
			// Lagged regularization form
			args["solver"]["advanced"]["lagged_regularization_weight"],
			args["solver"]["advanced"]["lagged_regularization_iterations"],
			// Augmented lagrangian form
			obstacle.ndof(), args["constraints"]["hard"], args["constraints"]["soft"],
			// Contact form
			args["contact"]["enabled"], args["contact"]["periodic"].get<bool>() ? periodic_collision_mesh : collision_mesh, args["contact"]["dhat"],
			avg_mass, args["contact"]["use_convergent_formulation"] ? bool(args["contact"]["use_area_weighting"]) : false,
			args["contact"]["use_convergent_formulation"] ? bool(args["contact"]["use_improved_max_operator"]) : false,
			args["contact"]["use_convergent_formulation"] ? bool(args["contact"]["use_physical_barrier"]) : false,
			args["solver"]["contact"]["barrier_stiffness"],
			args["solver"]["contact"]["initial_barrier_stiffness"],
			args["solver"]["contact"]["CCD"]["broad_phase"],
			args["solver"]["contact"]["CCD"]["tolerance"],
			args["solver"]["contact"]["CCD"]["max_iterations"],
			optimization_enabled == solver::CacheLevel::Derivatives,
			// Smooth Contact Form
			args["contact"]["use_gcp_formulation"],
			args["contact"]["alpha_t"],
			args["contact"]["alpha_n"],
			args["contact"]["use_adaptive_dhat"],
			args["contact"]["min_distance_ratio"],
			// Normal Adhesion Form
			args["contact"]["adhesion"]["adhesion_enabled"],
			args["contact"]["adhesion"]["dhat_p"],
			args["contact"]["adhesion"]["dhat_a"],
			args["contact"]["adhesion"]["adhesion_strength"],
			// Tangential Adhesion Form
			args["contact"]["adhesion"]["tangential_adhesion_coefficient"],
			args["contact"]["adhesion"]["epsa"],
			args["solver"]["contact"]["tangential_adhesion_iterations"],
			// Homogenization
			macro_strain_constraint,
			// Periodic contact
			args["contact"]["periodic"], periodic_collision_mesh_to_basis, periodic_bc,
			// Friction form
			args["contact"]["friction_coefficient"],
			args["contact"]["epsv"],
			args["solver"]["contact"]["friction_iterations"],
			// Rayleigh damping form
			args["solver"]["rayleigh_damping"]);

		for (const auto &form : forms)
			form->set_output_dir(output_dir);

		if (solve_data.contact_form != nullptr)
			solve_data.contact_form->save_ccd_debug_meshes = args["output"]["advanced"]["save_ccd_debug_meshes"];
	}

	void NonlinearElasticTransientVarForm::init_solve(Eigen::MatrixXd &sol, const double t)
	{
		assert(sol.cols() == 1);
		assert(!problem->is_scalar()); // tensor

		// FIXME
		//  if (optimization_enabled != solver::CacheLevel::None)
		//  {
		//  	if (initial_sol_update.size() == ndof())
		//  		sol = initial_sol_update;
		//  	else
		//  		initial_sol_update = sol;
		//  }

		// --------------------------------------------------------------------
		// Check for initial intersections
		if (is_contact_enabled())
		{
			POLYFEM_SCOPED_TIMER("Check for initial intersections");

			const Eigen::MatrixXd displaced = collision_mesh.displace_vertices(
				utils::unflatten(sol, mesh->dimension()));

			if (ipc::has_intersections(collision_mesh, displaced, ipc::create_broad_phase(args["solver"]["contact"]["CCD"]["broad_phase"])))
			{
				OBJWriter::write(
					resolve_output_path("intersection.obj"), displaced,
					collision_mesh.edges(), collision_mesh.faces());
				log_and_throw_error("Unable to solve, initial solution has intersections!");
			}
		}

		// --------------------------------------------------------------------

		{
			POLYFEM_SCOPED_TIMER("Initialize time integrator");
			solve_data.time_integrator = ImplicitTimeIntegrator::construct_time_integrator(args["time"]["integrator"]);

			Eigen::MatrixXd solution, velocity, acceleration;
			initial_solution(solution); // Reload this because we need all previous solutions
			solution.col(0) = sol;      // Make sure the current solution is the same as `sol`
			assert(solution.rows() == sol.size());
			initial_velocity(velocity);
			assert(velocity.rows() == sol.size());
			initial_acceleration(acceleration);
			assert(acceleration.rows() == sol.size());

			if (optimization_enabled != solver::CacheLevel::None)
			{
				if (initial_vel_update.size() == ndof())
					velocity = initial_vel_update;
				else
					initial_vel_update = velocity;
			}

			solve_data.time_integrator->init(solution, velocity, acceleration, dt);
		}
		assert(solve_data.time_integrator != nullptr);

		// --------------------------------------------------------------------
		// Initialize forms

		// --------------------------------------------------------------------
		// Initialize nonlinear problems

		const int ndof = n_bases * mesh->dimension();
		solve_data.nl_problem = std::make_shared<NLProblem>(
			ndof, periodic_bc, t, forms, solve_data.al_form,
			polysolve::linear::Solver::create(args["solver"]["linear"], logger()));
		solve_data.nl_problem->init(sol);
		solve_data.nl_problem->update_quantities(t, sol);
		// --------------------------------------------------------------------

		stats.solver_info = json::array();
	}

} // namespace polyfem::varform

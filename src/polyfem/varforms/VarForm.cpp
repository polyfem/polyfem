#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/varforms/VarFormUtils.hpp>

#include <polyfem/basis/LagrangeBasis2d.hpp>
#include <polyfem/basis/LagrangeBasis3d.hpp>
#include <polyfem/basis/SplineBasis2d.hpp>
#include <polyfem/basis/SplineBasis3d.hpp>
#include <polyfem/basis/barycentric/MVPolygonalBasis2d.hpp>
#include <polyfem/basis/barycentric/WSPolygonalBasis2d.hpp>
#include <polyfem/basis/PolygonalBasis2d.hpp>
#include <polyfem/basis/PolygonalBasis3d.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/refinement/APriori.hpp>
#include <polyfem/time_integrator/BDF.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Jacobian.hpp>

#include <igl/Timer.h>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/StringUtils.hpp>

#include <fstream>
#include <limits>

#include <spdlog/fmt/fmt.h>

namespace polyfem::varform
{
	bool VarForm::read_initial_x_from_file(
		const std::string &state_path,
		const std::string &x_name,
		const bool reorder,
		const Eigen::VectorXi &in_node_to_node,
		const int dim,
		Eigen::MatrixXd &x)
	{
		if (state_path.empty())
			return false;

		if (!io::read_matrix(state_path, x_name, x))
		{
			logger().debug("Unable to read initial {} from file ({})", x_name, state_path);
			return false;
		}

		if (reorder)
		{
			const int ndof = in_node_to_node.size() * dim;
			x.topRows(ndof) = utils::reorder_matrix(x.topRows(ndof), in_node_to_node, -1, dim);
		}

		return true;
	}

	namespace
	{

		bool should_use_iso_parametric(const mesh::Mesh &mesh, const json &args)
		{
			if (mesh.has_poly())
				return true;

			if (args["space"]["basis_type"] == "Bernstein")
				return false;

			if (args["space"]["basis_type"] == "Spline")
				return true;

			if (mesh.is_rational())
				return false;

			if (args["space"]["use_p_ref"])
				return false;

			if (args["boundary_conditions"]["periodic_boundary"]["enabled"].get<bool>())
				return false;

			if (mesh.orders().size() <= 0)
			{
				if (args["space"]["discr_order"] == 1)
					return true;
				return args["space"]["advanced"]["isoparametric"];
			}

			if (mesh.orders().minCoeff() != mesh.orders().maxCoeff())
				return false;

			if (args["space"]["discr_order"] == mesh.orders().minCoeff())
				return true;

			return args["space"]["advanced"]["isoparametric"];
		}

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

	QuadratureOrders VarForm::n_boundary_samples() const
	{
		using assembler::AssemblerUtils;
		const int n_b_samples_j = args["space"]["advanced"]["n_boundary_samples"];
		const int gdiscr_order = mesh_->orders().size() <= 0 ? 1 : mesh_->orders().maxCoeff();
		const int discr_order = std::max(disc_orders.maxCoeff(), gdiscr_order);

		const int n_b_samples = std::max(n_b_samples_j, AssemblerUtils::quadrature_order("Mass", discr_order, AssemblerUtils::BasisType::POLY, mesh_->dimension()));
		return {{n_b_samples, n_b_samples}};
	}

	void VarForm::reset()
	{
		// FIXME check subclasses
		stats.reset();
		timings = io::OutRuntimeData();
		output_sampler_initialized_ = false;
		prepared_ = false;
		rhs_assembler = nullptr;
		problem = nullptr;
		assembler = nullptr;
		mass_matrix_assembler = nullptr;
		pure_mass_matrix_assembler = nullptr;
		bases.clear();
		geom_bases_.clear();
		polys.clear();
		polys_3d.clear();
		poly_edge_to_data.clear();
		mesh_nodes = nullptr;
		geom_mesh_nodes = nullptr;
		in_node_to_node.resize(0);
		in_primitive_to_primitive.resize(0);
		disc_orders.resize(0);
		disc_ordersq.resize(0);
		ass_vals_cache.init_empty();
		mass_ass_vals_cache.init_empty(true);
		pure_mass_ass_vals_cache.init_empty(true);
		mass.resize(0, 0);
		pure_mass.resize(0, 0);
		rhs.resize(0, 0);
		avg_mass = 0;
		boundary_nodes.clear();
		total_local_boundary.clear();
		local_boundary.clear();
		local_neumann_boundary.clear();
		local_pressure_boundary.clear();
		local_pressure_cavity.clear();
		dirichlet_nodes.clear();
		dirichlet_nodes_position.clear();
		neumann_nodes.clear();
		neumann_nodes_position.clear();
		n_bases = 0;
		n_geom_bases = 0;
		t0 = 0;
		time_steps = 0;
		dt = 0;
		time_callback = nullptr;
		mesh_ = nullptr;
	}

	void VarForm::initial_solution(Eigen::MatrixXd &solution) const
	{
		assert(rhs_assembler != nullptr);

		const bool was_solution_loaded = read_initial_x_from_file(
			resolve_input_path(args["input"]["data"]["state"]), "u",
			args["input"]["data"]["reorder"], in_node_to_node,
			mesh_->dimension(), solution);

		if (!was_solution_loaded)
		{
			if (problem->is_time_dependent())
				rhs_assembler->initial_solution(solution);
			else
			{
				solution.resize(rhs.size(), 1);
				solution.setZero();
			}
		}
	}

	void VarForm::init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
	{
		reset();

		this->units = units;
		this->args = args;

		if (utils::is_param_valid(args, "root_path"))
			root_path = args["root_path"].get<std::string>();
		else
			root_path = "";

		this->output_path = out_path;
		output_sampler_initialized_ = false;
	}

	void VarForm::set_mesh(std::unique_ptr<mesh::Mesh> mesh, const double loading_mesh_time)
	{
		mesh_ = std::move(mesh);
		timings.loading_mesh_time = loading_mesh_time;
		output_sampler_initialized_ = false;
		prepared_ = false;
		if (!mesh_)
			return;

		load_mesh(*mesh_, args);
	}

	void VarForm::prepare()
	{
		if (prepared_)
			return;
		if (!mesh_)
			log_and_throw_error("Load the mesh first!");

		mesh_->prepare_mesh();
		stats.compute_mesh_stats(*mesh_);
		build_basis(*mesh_, should_use_iso_parametric(*mesh_, args), args);
		assemble_rhs(*mesh_, args);
		assemble_mass_mat(*mesh_, args);
		prepared_ = true;
	}

	void VarForm::build_basis(mesh::Mesh &mesh, const bool iso_parametric, const json &args)
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

		boundary_nodes.clear();
		dirichlet_nodes.clear();
		neumann_nodes.clear();
		dirichlet_nodes_position.clear();
		neumann_nodes_position.clear();
		total_local_boundary.clear();
		local_boundary.clear();
		local_neumann_boundary.clear();
		local_pressure_boundary.clear();
		local_pressure_cavity.clear();
		std::map<int, basis::InterfaceData> poly_edge_to_data_geom;

		const int quadrature_order = args["space"]["advanced"]["quadrature_order"].get<int>();
		const int mass_quadrature_order = args["space"]["advanced"]["mass_quadrature_order"].get<int>();

		// shape optimization needs continuous geometric basis
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
				n_bases = basis::SplineBasis2d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, bases, local_boundary, poly_edge_to_data);
			}
			else
			{
				if (!iso_parametric)
					n_geom_bases = basis::LagrangeBasis2d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, geom_disc_orders, false, false, has_polys, !use_continuous_gbasis, use_corner_quadrature, geom_bases_, local_boundary, poly_edge_to_data_geom, geom_mesh_nodes);

				n_bases = basis::LagrangeBasis2d::build_bases(tmp_mesh, assembler->name(), quadrature_order, mass_quadrature_order, disc_orders, args["space"]["basis_type"] == "Bernstein", args["space"]["basis_type"] == "Serendipity", has_polys, false, use_corner_quadrature, bases, local_boundary, poly_edge_to_data, mesh_nodes);
			}
		}

		timer.stop();

		build_polygonal_basis(mesh);

		if (n_geom_bases == 0)
			n_geom_bases = n_bases;

		total_local_boundary.clear();
		for (const auto &lb : local_boundary)
			total_local_boundary.emplace_back(lb);

		if (args["space"]["advanced"]["count_flipped_els"])
			stats.count_flipped_elements(mesh, geom_bases());

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

		// FIXME ME
		std::vector<basis::ElementBases> empty_pressure_bases;
		std::vector<int> empty_pressure_boundary_nodes;
		problem->setup_bc(
			mesh, n_bases,
			bases, geom_bases(), empty_pressure_bases,
			local_boundary,
			boundary_nodes,
			local_neumann_boundary,
			local_pressure_boundary,
			local_pressure_cavity,
			empty_pressure_boundary_nodes,
			dirichlet_nodes, neumann_nodes);

		rebuild_node_positions(bases, dirichlet_nodes, dirichlet_nodes_position);
		rebuild_node_positions(bases, neumann_nodes, neumann_nodes_position);

		const auto &current_bases = geom_bases();
		const int n_samples = 10;
		stats.compute_mesh_size(mesh, current_bases, n_samples, args["output"]["advanced"]["curved_mesh_size"]);

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
			ass_vals_cache.init(mesh.is_volume(), bases, current_bases);
			mass_ass_vals_cache.init(mesh.is_volume(), bases, current_bases, true);
			pure_mass_ass_vals_cache.init(mesh.is_volume(), bases, current_bases, true);

			logger().info(" took {}s", timer.getElapsedTime());
		}
		else
		{
			ass_vals_cache.init_empty();
			mass_ass_vals_cache.init_empty(true);
			pure_mass_ass_vals_cache.init_empty(true);
		}
	}

	void VarForm::build_fe_space(
		mesh::Mesh &mesh,
		const bool iso_parametric,
		const Eigen::VectorXi &disc_orders,
		const std::string &basis_type,
		const std::string &poly_basis_type,
		const assembler::Assembler &space_assembler,
		const int value_dim,
		const int quadrature_order,
		const int mass_quadrature_order,
		const bool use_corner_quadrature,
		const int n_harmonic_samples,
		const int integral_constraints,
		FESpace &space,
		VarFormBoundaryState &boundary)
	{
		using namespace mesh;

		const std::string space_assembler_name = space_assembler.name();

		space.reset();
		boundary.reset();

		space.value_dim = value_dim;

		space.bases = std::make_shared<std::vector<basis::ElementBases>>();
		space.geometry = std::make_shared<GeometryMapping>();

		space.disc_orders = disc_orders;
		space.disc_ordersq = disc_orders;

		Eigen::MatrixXi geom_disc_orders;
		if (!iso_parametric)
		{
			if (mesh.orders().size() <= 0)
			{
				geom_disc_orders.resizeLike(space.disc_orders);
				geom_disc_orders.setConstant(1);
			}
			else
				geom_disc_orders = mesh.orders();

			space.geometry->bases = std::make_shared<std::vector<basis::ElementBases>>();
			space.geometry->disc_orders = geom_disc_orders;
		}

		Eigen::MatrixXi geom_disc_ordersq = geom_disc_orders;

		logger().info("Building {} basis...", (iso_parametric ? "isoparametric" : "not isoparametric"));

		igl::Timer timer;
		timer.start();

		const bool has_polys = mesh.has_poly();
		std::map<int, basis::InterfaceData> poly_edge_to_data_geom;

		const bool use_continuous_gbasis = true;

		if (mesh.is_volume())
		{
			const Mesh3D &tmp_mesh = dynamic_cast<const Mesh3D &>(mesh);

			if (basis_type == "Spline")
			{
				space.n_bases = basis::SplineBasis3d::build_bases(
					tmp_mesh, space_assembler_name, quadrature_order, mass_quadrature_order,
					*space.bases, boundary.local_boundary, space.poly_edge_to_data);
			}
			else
			{
				if (!iso_parametric)
					space.geometry->n_bases = basis::LagrangeBasis3d::build_bases(
						tmp_mesh, space_assembler_name, quadrature_order, mass_quadrature_order,
						geom_disc_orders, geom_disc_ordersq, false, false, has_polys,
						!use_continuous_gbasis, use_corner_quadrature,
						*space.geometry->bases, boundary.local_boundary, poly_edge_to_data_geom,
						space.geometry->mesh_nodes);

				space.n_bases = basis::LagrangeBasis3d::build_bases(
					tmp_mesh, space_assembler_name, quadrature_order, mass_quadrature_order,
					space.disc_orders, space.disc_ordersq,
					basis_type == "Bernstein",
					basis_type == "Serendipity",
					has_polys, false, use_corner_quadrature,
					*space.bases, boundary.local_boundary, space.poly_edge_to_data, space.mesh_nodes);
			}
		}
		else
		{
			const Mesh2D &tmp_mesh = dynamic_cast<const Mesh2D &>(mesh);

			if (basis_type == "Spline")
			{
				space.n_bases = basis::SplineBasis2d::build_bases(
					tmp_mesh, space_assembler_name, quadrature_order, mass_quadrature_order,
					*space.bases, boundary.local_boundary, space.poly_edge_to_data);
			}
			else
			{
				if (!iso_parametric)
					space.geometry->n_bases = basis::LagrangeBasis2d::build_bases(
						tmp_mesh, space_assembler_name, quadrature_order, mass_quadrature_order,
						geom_disc_orders, false, false, has_polys,
						!use_continuous_gbasis, use_corner_quadrature,
						*space.geometry->bases, boundary.local_boundary, poly_edge_to_data_geom,
						space.geometry->mesh_nodes);

				space.n_bases = basis::LagrangeBasis2d::build_bases(
					tmp_mesh, space_assembler_name, quadrature_order, mass_quadrature_order,
					space.disc_orders,
					basis_type == "Bernstein",
					basis_type == "Serendipity",
					has_polys, false, use_corner_quadrature,
					*space.bases, boundary.local_boundary, space.poly_edge_to_data, space.mesh_nodes);
			}
		}

		build_polygonal_basis(mesh, poly_basis_type, space_assembler,
							  iso_parametric,
							  quadrature_order,
							  mass_quadrature_order,
							  n_harmonic_samples,
							  integral_constraints,
							  space,
							  boundary);

		if (iso_parametric)
			space.geometry->init_from_fe_space(space);
		else
		{
			assert(space.geometry->bases);
			assert(space.geometry->n_bases > 0);
		}

		boundary.total_local_boundary.clear();
		for (const auto &lb : boundary.local_boundary)
			boundary.total_local_boundary.emplace_back(lb);

		{
			igl::Timer timer2;
			logger().debug("Building node mapping...");
			timer2.start();
			build_node_mapping(mesh, basis_type, space, space.space_in_node_to_node, space.space_in_primitive_to_primitive);
			timer2.stop();
			logger().debug("Done (took {}s)", timer2.getElapsedTime());
		}

		logger().info("n_bases {}", space.n_bases);

		timings.building_basis_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.building_basis_time);

		logger().info("n bases: {}", space.n_bases);
	}

	void VarForm::build_polygonal_basis(const mesh::Mesh &mesh)
	{
		rhs.resize(0, 0);

		if (poly_edge_to_data.empty() && polys.empty())
		{
			timings.computing_poly_basis_time = 0;
			return;
		}

		igl::Timer timer;
		timer.start();
		logger().info("Computing polygonal basis...");

		int new_bases = 0;
		const int dim = assembler->is_tensor() ? mesh.dimension() : 1;
		if (iso_parametric)
		{
			if (mesh.is_volume())
			{
				if (args["space"]["poly_basis_type"] == "MeanValue" || args["space"]["poly_basis_type"] == "Wachspress")
					logger().error("Barycentric bases not supported in 3D");

				const auto *linear_assembler = dynamic_cast<assembler::LinearAssembler *>(assembler.get());
				assert(linear_assembler);
				new_bases = basis::PolygonalBasis3d::build_bases(
					*linear_assembler,
					args["space"]["advanced"]["n_harmonic_samples"],
					dynamic_cast<const mesh::Mesh3D &>(mesh),
					n_bases,
					args["space"]["advanced"]["quadrature_order"],
					args["space"]["advanced"]["mass_quadrature_order"],
					args["space"]["advanced"]["integral_constraints"],
					bases,
					bases,
					poly_edge_to_data,
					polys_3d);
			}
			else
			{
				const mesh::Mesh2D &mesh_2d = dynamic_cast<const mesh::Mesh2D &>(mesh);
				if (args["space"]["poly_basis_type"] == "MeanValue")
				{
					new_bases = basis::MVPolygonalBasis2d::build_bases(
						assembler->name(), dim, mesh_2d, n_bases,
						args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						bases, local_boundary, polys);
				}
				else if (args["space"]["poly_basis_type"] == "Wachspress")
				{
					new_bases = basis::WSPolygonalBasis2d::build_bases(
						assembler->name(), dim, mesh_2d, n_bases,
						args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						bases, local_boundary, polys);
				}
				else
				{
					const auto *linear_assembler = dynamic_cast<assembler::LinearAssembler *>(assembler.get());
					assert(linear_assembler);
					new_bases = basis::PolygonalBasis2d::build_bases(
						*linear_assembler,
						args["space"]["advanced"]["n_harmonic_samples"],
						mesh_2d,
						n_bases,
						args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						args["space"]["advanced"]["integral_constraints"],
						bases,
						bases,
						poly_edge_to_data,
						polys);
				}
			}
		}
		else
		{
			if (mesh.is_volume())
			{
				if (args["space"]["poly_basis_type"] == "MeanValue" || args["space"]["poly_basis_type"] == "Wachspress")
					log_and_throw_error("Barycentric bases not supported in 3D");

				const auto *linear_assembler = dynamic_cast<assembler::LinearAssembler *>(assembler.get());
				assert(linear_assembler);
				new_bases = basis::PolygonalBasis3d::build_bases(
					*linear_assembler,
					args["space"]["advanced"]["n_harmonic_samples"],
					dynamic_cast<const mesh::Mesh3D &>(mesh),
					n_bases,
					args["space"]["advanced"]["quadrature_order"],
					args["space"]["advanced"]["mass_quadrature_order"],
					args["space"]["advanced"]["integral_constraints"],
					bases,
					geom_bases_,
					poly_edge_to_data,
					polys_3d);
			}
			else
			{
				const mesh::Mesh2D &mesh_2d = dynamic_cast<const mesh::Mesh2D &>(mesh);
				if (args["space"]["poly_basis_type"] == "MeanValue")
				{
					new_bases = basis::MVPolygonalBasis2d::build_bases(
						assembler->name(), dim, mesh_2d, n_bases,
						args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						bases, local_boundary, polys);
				}
				else if (args["space"]["poly_basis_type"] == "Wachspress")
				{
					new_bases = basis::WSPolygonalBasis2d::build_bases(
						assembler->name(), dim, mesh_2d, n_bases,
						args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						bases, local_boundary, polys);
				}
				else
				{
					const auto *linear_assembler = dynamic_cast<assembler::LinearAssembler *>(assembler.get());
					assert(linear_assembler);
					new_bases = basis::PolygonalBasis2d::build_bases(
						*linear_assembler,
						args["space"]["advanced"]["n_harmonic_samples"],
						mesh_2d,
						n_bases,
						args["space"]["advanced"]["quadrature_order"],
						args["space"]["advanced"]["mass_quadrature_order"],
						args["space"]["advanced"]["integral_constraints"],
						bases,
						geom_bases_,
						poly_edge_to_data,
						polys);
				}
			}
		}

		timer.stop();
		timings.computing_poly_basis_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.computing_poly_basis_time);

		n_bases += new_bases;
	}

	void VarForm::build_polygonal_basis(
		const mesh::Mesh &mesh,
		const std::string &poly_basis_type,
		const assembler::Assembler &space_assembler,
		bool iso_parametric,
		const int quadrature_order,
		const int mass_quadrature_order,
		const int n_harmonic_samples,
		const int integral_constraints,
		varform::FESpace &space,
		varform::VarFormBoundaryState &boundary)
	{
		if (space.poly_edge_to_data.empty() && space.polys.empty())
		{
			timings.computing_poly_basis_time = 0;
			return;
		}

		const std::string space_assembler_name = space_assembler.name();

		igl::Timer timer;
		timer.start();
		logger().info("Computing polygonal basis...");

		int new_bases = 0;
		const int dim = space_assembler.is_tensor() ? mesh.dimension() : 1;
		if (iso_parametric)
		{
			if (mesh.is_volume())
			{
				if (poly_basis_type == "MeanValue" || poly_basis_type == "Wachspress")
					log_and_throw_error("Barycentric bases not supported in 3D");

				const auto *linear_assembler = dynamic_cast<const assembler::LinearAssembler *>(&space_assembler);
				assert(linear_assembler);
				new_bases = basis::PolygonalBasis3d::build_bases(
					*linear_assembler,
					n_harmonic_samples,
					dynamic_cast<const mesh::Mesh3D &>(mesh),
					space.n_bases,
					quadrature_order,
					mass_quadrature_order,
					integral_constraints,
					*space.bases,
					*space.bases,
					space.poly_edge_to_data,
					space.polys_3d);
			}
			else
			{
				const mesh::Mesh2D &mesh_2d = dynamic_cast<const mesh::Mesh2D &>(mesh);
				if (poly_basis_type == "MeanValue")
				{
					new_bases = basis::MVPolygonalBasis2d::build_bases(
						space_assembler.name(), dim, mesh_2d, space.n_bases,
						quadrature_order,
						mass_quadrature_order,
						*space.bases, boundary.local_boundary, space.polys);
				}
				else if (poly_basis_type == "Wachspress")
				{
					new_bases = basis::WSPolygonalBasis2d::build_bases(
						space_assembler.name(), dim, mesh_2d, space.n_bases,
						quadrature_order,
						mass_quadrature_order,
						*space.bases, boundary.local_boundary, space.polys);
				}
				else
				{
					const auto *linear_assembler = dynamic_cast<const assembler::LinearAssembler *>(&space_assembler);
					assert(linear_assembler);
					new_bases = basis::PolygonalBasis2d::build_bases(
						*linear_assembler,
						n_harmonic_samples,
						mesh_2d,
						space.n_bases,
						quadrature_order,
						mass_quadrature_order,
						integral_constraints,
						*space.bases,
						*space.bases,
						space.poly_edge_to_data,
						space.polys);
				}
			}
		}
		else
		{
			assert(space.geometry);
			assert(space.geometry->bases);
			if (mesh.is_volume())
			{
				if (poly_basis_type == "MeanValue" || poly_basis_type == "Wachspress")
					log_and_throw_error("Barycentric bases not supported in 3D");

				const auto *linear_assembler = dynamic_cast<const assembler::LinearAssembler *>(&space_assembler);
				assert(linear_assembler);
				new_bases = basis::PolygonalBasis3d::build_bases(
					*linear_assembler,
					n_harmonic_samples,
					dynamic_cast<const mesh::Mesh3D &>(mesh),
					space.n_bases,
					quadrature_order,
					mass_quadrature_order,
					integral_constraints,
					*space.bases,
					*space.geometry->bases,
					space.poly_edge_to_data,
					space.polys_3d);
			}
			else
			{
				const mesh::Mesh2D &mesh_2d = dynamic_cast<const mesh::Mesh2D &>(mesh);
				if (poly_basis_type == "MeanValue")
				{
					new_bases = basis::MVPolygonalBasis2d::build_bases(
						space_assembler.name(), dim, mesh_2d, space.n_bases,
						quadrature_order,
						mass_quadrature_order,
						*space.bases, boundary.local_boundary, space.polys);
				}
				else if (poly_basis_type == "Wachspress")
				{
					new_bases = basis::WSPolygonalBasis2d::build_bases(
						space_assembler.name(), dim, mesh_2d, space.n_bases,
						quadrature_order,
						mass_quadrature_order,
						*space.bases, boundary.local_boundary, space.polys);
				}
				else
				{
					const auto *linear_assembler = dynamic_cast<const assembler::LinearAssembler *>(&space_assembler);
					assert(linear_assembler);
					new_bases = basis::PolygonalBasis2d::build_bases(
						*linear_assembler,
						n_harmonic_samples,
						mesh_2d,
						space.n_bases,
						quadrature_order,
						mass_quadrature_order,
						integral_constraints,
						*space.bases,
						*space.geometry->bases,
						space.poly_edge_to_data,
						space.polys);
				}
			}
		}

		timer.stop();
		timings.computing_poly_basis_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.computing_poly_basis_time);

		space.n_bases += new_bases;
	}

	void VarForm::solve(Eigen::MatrixXd &sol)
	{
		prepare();
		solve_problem(sol);
	}

	void VarForm::build_node_mapping(const mesh::Mesh &mesh, const json &args)
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

	void VarForm::build_node_mapping(
		const mesh::Mesh &mesh,
		const std::string &basis_type,
		const FESpace &space,
		Eigen::VectorXi &space_in_node_to_node,
		Eigen::VectorXi &space_in_primitive_to_primitive) const
	{
		space_in_node_to_node.resize(0);
		space_in_primitive_to_primitive.resize(0);

		if (basis_type == "Spline")
		{
			logger().warn("Node ordering disabled, it dosent work for splines!");
			return;
		}

		if (space.disc_orders.maxCoeff() >= 4 || space.disc_orders.maxCoeff() != space.disc_orders.minCoeff())
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

		if (!space.mesh_nodes)
		{
			logger().warn("Node ordering disabled, FE space does not expose mesh nodes!");
			return;
		}

		const int num_vertex_nodes = space.mesh_nodes->num_vertex_nodes();
		const int num_edge_nodes = space.mesh_nodes->num_edge_nodes();
		const int num_face_nodes = space.mesh_nodes->num_face_nodes();
		const int num_cell_nodes = space.mesh_nodes->num_cell_nodes();

		const int num_nodes = num_vertex_nodes + num_edge_nodes + num_face_nodes + num_cell_nodes;
		const long n_vertices = num_vertex_nodes;
		const int num_in_primitives = n_vertices + mesh.n_edges() + mesh.n_faces() + mesh.n_cells();
		const int num_primitives = mesh.n_vertices() + mesh.n_edges() + mesh.n_faces() + mesh.n_cells();

		igl::Timer timer;

		logger().trace("Building in-node to in-primitive mapping...");
		timer.start();
		Eigen::VectorXi in_node_to_in_primitive;
		Eigen::VectorXi in_node_offset;
		build_in_node_to_in_primitive(mesh, *space.mesh_nodes, in_node_to_in_primitive, in_node_offset);
		timer.stop();
		logger().trace("Done (took {}s)", timer.getElapsedTime());

		logger().trace("Building in-primitive to primitive mapping...");
		timer.start();
		bool ok = build_in_primitive_to_primitive(
			mesh, *space.mesh_nodes,
			mesh.in_ordered_vertices(),
			mesh.in_ordered_edges(),
			mesh.in_ordered_faces(),
			space_in_primitive_to_primitive);
		timer.stop();
		logger().trace("Done (took {}s)", timer.getElapsedTime());

		if (!ok)
		{
			space_in_node_to_node.resize(0);
			space_in_primitive_to_primitive.resize(0);
			return;
		}

		const auto &tmp = space.mesh_nodes->in_ordered_vertices();
		int max_tmp = -1;
		for (auto v : tmp)
			max_tmp = std::max(max_tmp, v);

		space_in_node_to_node.resize(max_tmp + 1);
		for (int i = 0; i < tmp.size(); ++i)
		{
			if (tmp[i] >= 0)
				space_in_node_to_node[tmp[i]] = i;
		}
	}

	void VarForm::assign_discr_orders(const json &discr_order, const mesh::Mesh &mesh, Eigen::VectorXi &disc_orders)
	{
		disc_orders.resize(mesh.n_elements());

		if (discr_order.is_number_integer())
		{
			disc_orders.setConstant(discr_order);
		}
		else if (discr_order.is_string())
		{
			const std::string discr_orders_path = utils::resolve_path(discr_order, root_path);
			Eigen::MatrixXi tmp;
			io::read_matrix(discr_orders_path, tmp);
			assert(tmp.size() == disc_orders.size());
			assert(tmp.cols() == 1);
			disc_orders = tmp;
		}
		else if (discr_order.is_array())
		{
			const auto b_discr_orders = discr_order;

			std::map<int, int> b_orders;
			for (size_t i = 0; i < b_discr_orders.size(); ++i)
			{
				assert(b_discr_orders[i]["id"].is_array() || b_discr_orders[i]["id"].is_number_integer());

				const int order = b_discr_orders[i]["order"];
				for (const int id : utils::json_as_array<int>(b_discr_orders[i]["id"]))
				{
					b_orders[id] = order;
					logger().trace("bid {}, discr {}", id, order);
				}
			}

			for (int e = 0; e < mesh.n_elements(); ++e)
			{
				const int bid = mesh.get_body_id(e);
				const auto order = b_orders.find(bid);
				if (order == b_orders.end())
				{
					logger().debug("Missing discretization order for body {}; using 1", bid);
					b_orders[bid] = 1;
					disc_orders[e] = 1;
				}
				else
				{
					disc_orders[e] = order->second;
				}
			}
		}
		else
		{
			logger().error("space/discr_order must be either a number a path or an array");
			throw std::runtime_error("invalid json");
		}
	}

	void VarForm::save_json(const Eigen::MatrixXd &solution) const
	{
		const std::string out_path = resolve_output_path(args["output"]["json"]);
		if (out_path.empty())
			return;

		std::ofstream file(out_path);
		if (!file.is_open())
		{
			logger().error("Unable to save simulation JSON to {}", out_path);
			return;
		}
		save_json(solution, file);
	}

	void VarForm::save_json_stats(
		const Eigen::MatrixXd &solution,
		const int n_auxiliary_bases,
		std::ostream &out) const
	{
		if (!mesh_)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (solution.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		logger().info("Saving json...");
		const int primary_size = n_bases * problem_dimension();
		const Eigen::MatrixXd stats_solution =
			solution.rows() >= primary_size
				? solution.topRows(primary_size).eval()
				: solution;

		nlohmann::json j;
		stats.save_json(
			args, n_bases, n_auxiliary_bases,
			stats_solution, *mesh_, disc_orders, disc_ordersq, *problem,
			timings, assembler ? assembler->name() : name(), iso_parametric,
			args["output"]["advanced"]["sol_at_node"], j);
		out << j.dump(4) << std::endl;
	}

	std::vector<io::OutputField> VarForm::common_output_fields(
		const io::OutputSample &sample,
		const Eigen::MatrixXd &solution,
		const io::OutputFieldOptions &options) const
	{
		std::vector<io::OutputField> fields;
		if (!mesh_ || !problem || solution.size() <= 0)
			return fields;

		const bool has_element_samples =
			sample.local_points.rows() > 0
			&& sample.local_points.rows() == sample.element_ids.size();
		if (!has_element_samples)
			return fields;

		const int dim = problem_dimension();
		const int output_rows = sample.points.rows() > 0 ? sample.points.rows() : sample.local_points.rows();
		const int primary_ndof = std::min<int>(solution.rows(), n_bases * dim);
		const Eigen::MatrixXd primary_solution = solution.topRows(primary_ndof);

		const auto sample_dof_values = [&](const Eigen::MatrixXd &dof_values, Eigen::MatrixXd &values, Eigen::MatrixXd *gradients = nullptr) {
			values.setZero(output_rows, dim);
			if (gradients)
				gradients->setZero(output_rows, dim * mesh_->dimension());

			for (int i = 0; i < sample.local_points.rows(); ++i)
			{
				const int element_id = sample.element_ids(i);
				if (element_id < 0)
					continue;

				Eigen::MatrixXd local_value, local_gradient;
				io::Evaluator::interpolate_at_local_vals(
					*mesh_, dim, bases, geom_bases(),
					element_id, sample.local_points.row(i), dof_values,
					local_value, local_gradient);
				values.row(i) = local_value;
				if (gradients)
					gradients->row(i) = local_gradient;
			}
		};

		if (problem->has_exact_sol() && sample.points.rows() == output_rows)
		{
			Eigen::MatrixXd exact;
			problem->exact(sample.points, sample.time, exact);
			if (exact.rows() == output_rows)
			{
				if (options.export_field("exact"))
					fields.push_back({"exact", exact, io::OutputField::Association::Point});
				if (options.export_field("error"))
				{
					Eigen::MatrixXd values;
					sample_dof_values(primary_solution, values);
					fields.push_back({"error", (values - exact).rowwise().norm(), io::OutputField::Association::Point});
				}
			}
		}

		const auto &paraview_options = args["output"]["paraview"]["options"];
		if ((paraview_options["nodes"] || (!options.fields.empty() && options.export_field("nodes")))
			&& sample.primitive_ids.size() == 0)
		{
			Eigen::MatrixXd dof_ids(primary_ndof, 1);
			dof_ids.col(0).setLinSpaced(primary_ndof, 0, primary_ndof - 1);
			Eigen::MatrixXd values;
			sample_dof_values(dof_ids, values);
			fields.push_back({"nodes", values, io::OutputField::Association::Point});
		}

		if ((paraview_options["jacobian_validity"] || (!options.fields.empty() && options.export_field("validity")))
			&& dim == mesh_->dimension()
			&& sample.primitive_ids.size() == 0)
		{
			const auto invalid_elements = utils::count_invalid(mesh_->dimension(), bases, geom_bases(), primary_solution);
			Eigen::MatrixXd validity = Eigen::MatrixXd::Zero(output_rows, 1);
			for (int i = 0; i < sample.element_ids.size(); ++i)
				validity(i) = std::find(invalid_elements.begin(), invalid_elements.end(), sample.element_ids(i)) != invalid_elements.end();
			fields.push_back({"validity", validity, io::OutputField::Association::Point});
		}

		return fields;
	}

	std::vector<int> VarForm::primitive_to_node() const
	{
		const auto &nodes = iso_parametric ? mesh_nodes : geom_mesh_nodes;
		if (!nodes)
			return {};

		auto indices = nodes->primitive_to_node();
		indices.resize(mesh_->n_vertices());
		return indices;
	}

	std::vector<int> VarForm::node_to_primitive() const
	{
		auto p2n = primitive_to_node();
		std::vector<int> indices;
		indices.resize(n_geom_bases);
		for (int i = 0; i < p2n.size(); i++)
			indices[p2n[i]] = i;
		return indices;
	}

	void VarForm::assemble_rhs(const mesh::Mesh &mesh, const json &args)
	{
		igl::Timer timer;
		json p_params = {};
		p_params["formulation"] = assembler->name();
		p_params["root_path"] = root_path;
		{
			RowVectorNd min, max, delta;
			mesh.bounding_box(min, max);
			delta = (max - min) / 2. + min;
			if (mesh.is_volume())
				p_params["bbox_center"] = {delta(0), delta(1), delta(2)};
			else
				p_params["bbox_center"] = {delta(0), delta(1)};
		}
		problem->set_parameters(p_params, root_path);

		rhs.resize(0, 0);

		timer.start();
		logger().info("Assigning rhs...");

		build_rhs_assembler();
		assert(rhs_assembler != nullptr);
		rhs_assembler->assemble(mass_matrix_assembler->density(), rhs);
		rhs *= -1;

		timings.assigning_rhs_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assigning_rhs_time);
	}

	void VarForm::assemble_mass_mat(const mesh::Mesh &mesh, const json &args)
	{
		if (!problem->is_time_dependent())
		{
			avg_mass = 1;
			timings.assembling_mass_mat_time = 0;
			if (!assembler->is_linear())
				pure_mass_matrix_assembler->assemble(mesh.is_volume(), n_bases, bases, geom_bases(), pure_mass_ass_vals_cache, 0, pure_mass, true);
			return;
		}

		mass.resize(0, 0);

		igl::Timer timer;
		timer.start();
		logger().info("Assembling mass mat...");

		mass_matrix_assembler->assemble(mesh.is_volume(), n_bases, bases, geom_bases(), mass_ass_vals_cache, 0, mass, true);
		if (!assembler->is_linear())
			pure_mass_matrix_assembler->assemble(mesh.is_volume(), n_bases, bases, geom_bases(), pure_mass_ass_vals_cache, 0, pure_mass, true);

		assert(mass.size() > 0);

		avg_mass = 0;
		for (int k = 0; k < mass.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(mass, k); it; ++it)
			{
				assert(it.col() == k);
				avg_mass += it.value();
			}
		}

		avg_mass /= mass.rows();
		logger().info("average mass {}", avg_mass);

		if (args["solver"]["advanced"]["lump_mass_matrix"])
			mass = utils::lump_matrix(mass);

		timer.stop();
		timings.assembling_mass_mat_time = timer.getElapsedTime();
		logger().info(" took {}s", timings.assembling_mass_mat_time);

		stats.nn_zero = mass.nonZeros();
		stats.num_dofs = mass.rows();
		stats.mat_size = (long long)mass.rows() * (long long)mass.cols();
		logger().info("sparsity: {}/{}", stats.nn_zero, stats.mat_size);
	}

	void VarForm::set_materials(assembler::Assembler &assembler) const
	{
		assert(mesh_ != nullptr);
		const int size = this->assembler && (this->assembler->is_tensor() || this->assembler->is_fluid()) ? mesh_->dimension() : 1;
		assembler.set_size(size);

		if (!utils::is_param_valid(args, "materials"))
			return;

		std::vector<int> body_ids(mesh_->n_elements());
		for (int i = 0; i < mesh_->n_elements(); ++i)
			body_ids[i] = mesh_->get_body_id(i);

		assembler.set_materials(body_ids, args["materials"], units, root_path);
	}

	void VarForm::build_rhs_assembler()
	{
		json rhs_solver_params = args["solver"]["linear"];
		if (!rhs_solver_params.contains("Pardiso"))
			rhs_solver_params["Pardiso"] = {};
		rhs_solver_params["Pardiso"]["mtype"] = -2;

		const int size = problem->is_scalar() ? 1 : mesh_->dimension();

		rhs_assembler = std::make_shared<assembler::RhsAssembler>(
			*assembler, *mesh_, nullptr,
			dirichlet_nodes, neumann_nodes,
			dirichlet_nodes_position, neumann_nodes_position,
			n_bases, size, bases, geom_bases(), mass_ass_vals_cache, *problem,
			args["space"]["advanced"]["bc_method"],
			rhs_solver_params);
	}

	void VarForm::load_mesh(const mesh::Mesh &mesh, const json &args)
	{
		set_materials(*assembler);
		set_materials(*mass_matrix_assembler);
		pure_mass_matrix_assembler->set_size(mass_matrix_assembler->size());

		problem->init(mesh);
	}

	void VarForm::ensure_output_sampler() const
	{
		if (output_sampler_initialized_)
			return;

		const io::OutputSpace space = output_space();
		if (space.mesh)
		{
			output_geometry_.init_sampler(*space.mesh, args["output"]["paraview"]["vismesh_rel_area"]);
			output_geometry_.build_grid(*space.mesh, args["output"]["advanced"]["sol_on_grid"]);
		}
		output_sampler_initialized_ = true;
	}

	io::OutGeometryData::ExportOptions VarForm::export_options(const io::OutputSpace &space) const
	{
		return io::OutGeometryData::ExportOptions(
			args,
			space.mesh->is_linear(),
			space.mesh->has_prism(),
			problem_dimension() == 1);
	}

	io::OutputFieldFunction VarForm::output_field_function(const Eigen::MatrixXd &solution, const io::OutGeometryData::ExportOptions &opts) const
	{
		return [this, &solution, fields = opts.fields](const io::OutputSample &sample) {
			return output_fields(
				sample, solution,
				io::OutputFieldOptions{
					sample.requested_fields.empty() ? fields : sample.requested_fields});
		};
	}

	void VarForm::export_data(const Eigen::MatrixXd &solution) const
	{
		const io::OutputSpace space = output_space();
		if (!space.mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (solution.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		ensure_output_sampler();

		const std::string vis_mesh_path = resolve_output_path(args["output"]["paraview"]["file_name"]);
		const bool has_time = args.contains("time") && !args["time"].is_null();
		double tend = has_time ? args["time"]["tend"].get<double>() : 1.0;
		double dt = 1;
		if (has_time)
			dt = args["time"]["dt"];

		const auto opts = export_options(space);
		output_geometry_.export_data(
			space,
			output_field_function(solution, opts),
			has_time,
			tend, dt,
			opts,
			vis_mesh_path);

		const std::string solution_path = resolve_output_path(args["output"]["data"]["solution"]);
		if (!solution_path.empty())
		{
			const int dim = problem_dimension();
			const int primary_ndof = std::min<int>(solution.rows(), n_bases * dim);
			const Eigen::MatrixXd primary_solution = solution.topRows(primary_ndof);
			if (opts.reorder_output && in_node_to_node.size() > 0)
			{
				const Eigen::MatrixXd nodal_solution = utils::unflatten(primary_solution, dim);
				Eigen::MatrixXd reordered = Eigen::MatrixXd::Zero(nodal_solution.rows(), nodal_solution.cols());
				for (int input_node = 0; input_node < in_node_to_node.size(); ++input_node)
				{
					const int node = in_node_to_node(input_node);
					if (node >= 0 && node < nodal_solution.rows() && input_node < reordered.rows())
						reordered.row(input_node) = nodal_solution.row(node);
				}
				io::write_matrix(solution_path, reordered);
			}
			else
			{
				io::write_matrix(solution_path, primary_solution);
			}
		}

		const std::string nodes_path = resolve_output_path(args["output"]["data"]["nodes"]);
		if (!nodes_path.empty())
		{
			Eigen::MatrixXd nodes = Eigen::MatrixXd::Zero(n_bases, mesh_->dimension());
			for (const basis::ElementBases &element_bases : bases)
				for (const basis::Basis &basis : element_bases.bases)
					for (const auto &global : basis.global())
						nodes.row(global.index) = global.node;
			io::write_matrix(nodes_path, nodes);
		}

		const std::string stress_path = resolve_output_path(args["output"]["data"]["stress_mat"]);
		const std::string mises_path = resolve_output_path(args["output"]["data"]["mises"]);
		if ((!stress_path.empty() || !mises_path.empty()) && assembler)
		{
			Eigen::MatrixXd stress;
			Eigen::VectorXd mises;
			io::Evaluator::compute_stress_at_quadrature_points(
				*mesh_, problem->is_scalar(), bases, geom_bases(),
				disc_orders, disc_ordersq, *assembler, solution, tend,
				stress, mises);
			if (!stress_path.empty())
				io::write_matrix(stress_path, stress);
			if (!mises_path.empty())
				io::write_matrix(mises_path, mises);
		}
	}

	io::OutputSpace VarForm::output_space() const
	{
		Eigen::VectorXi output_orders = disc_orders;
		if (mesh_ && disc_ordersq.size() == disc_orders.size())
		{
			for (int e = 0; e < output_orders.size(); ++e)
			{
				if (mesh_->is_prism(e))
					output_orders(e) = std::max(disc_orders(e), disc_ordersq(e));
			}
		}

		return {
			mesh_.get(),
			&geom_bases(),
			output_orders,
			&polys,
			&polys_3d,
			&total_local_boundary,
			nullptr,
			nullptr,
			&dirichlet_nodes,
			&dirichlet_nodes_position};
	}

	int VarForm::problem_dimension() const
	{
		if (!problem)
			return 0;
		if (problem->is_scalar())
			return 1;
		return mesh_ ? mesh_->dimension() : 0;
	}

	std::shared_ptr<time_integrator::BDF> VarForm::make_bdf_time_integrator() const
	{
		const json &config = args["time"]["integrator"];
		const std::string type = config.is_object() ? config.at("type").get<std::string>() : config.get<std::string>();

		if (type == "ImplicitEuler" || type == "implict_euler")
			return std::make_shared<time_integrator::BDF>();

		if (!utils::StringUtils::startswith(type, "BDF"))
			log_and_throw_error("First-order transient formulations require ImplicitEuler or BDF, got {}.", type);

		auto integrator = std::make_shared<time_integrator::BDF>(
			type == "BDF" ? 1 : std::stoi(type.substr(3)));
		if (config.is_object() && config.contains("steps"))
			integrator->set_parameters(config);
		return integrator;
	}

	void VarForm::save_step_state(
		const double t0,
		const double dt,
		const int t,
		const time_integrator::ImplicitTimeIntegrator *time_integrator,
		const bool rest_mesh_written) const
	{
		const int global_t = output_file_index(t);
		const std::string state_path = resolve_output_path(fmt::format(args["output"]["data"]["state"], global_t));
		if (!state_path.empty() && time_integrator)
			time_integrator->save_state(state_path);

		save_restart_json(t0, dt, t, rest_mesh_written);
	}

	void VarForm::save_timestep(const double time, const int t, const double t0, const double dt, const Eigen::MatrixXd &solution) const
	{
		const io::OutputSpace space = output_space();
		if (!space.mesh || !args["output"]["advanced"]["save_time_sequence"])
			return;
		const int global_t = output_file_index(t);
		if (global_t % args["output"]["paraview"]["skip_frame"].get<int>())
			return;

		ensure_output_sampler();

		logger().trace("Saving VTU...");
		const std::string step_name = args["output"]["advanced"]["timestep_prefix"];
		const auto opts = export_options(space);
		output_geometry_.save_vtu(
			resolve_output_path(fmt::format(step_name + "{:d}.vtu", global_t)),
			space, output_field_function(solution, opts), time, dt,
			opts);

		output_geometry_.save_pvd(
			resolve_output_path(args["output"]["paraview"]["file_name"]),
			[step_name](int i) { return fmt::format(step_name + "{:d}.vtm", i); },
			global_t, t0, dt, args["output"]["paraview"]["skip_frame"].get<int>());
	}

	void VarForm::save_subsolve(const int i, const int t, const Eigen::MatrixXd &solution) const
	{
		const io::OutputSpace space = output_space();
		if (!space.mesh || !args["output"]["advanced"]["save_solve_sequence_debug"].get<bool>())
			return;

		const bool has_time = args.contains("time") && !args["time"].is_null();
		double dt = 1;
		if (has_time)
			dt = args["time"]["dt"];

		ensure_output_sampler();
		const auto opts = export_options(space);
		output_geometry_.save_vtu(
			resolve_output_path(fmt::format("solve_{:d}.vtu", i)),
			space, output_field_function(solution, opts), t, dt,
			opts);
	}

	void VarForm::notify_time_step(const int t) const
	{
		if (time_callback)
			time_callback(t, time_steps, t0 + dt * t, t0 + dt * time_steps);
	}

	void VarForm::save_restart_json(const double t0, const double dt, const int t, const bool rest_mesh_written) const
	{
		const std::string restart_json_path = args["output"]["restart_json"];
		if (restart_json_path.empty())
			return;

		const int global_t = output_file_index(t);

		json restart_json;
		restart_json["root_path"] = root_path;
		restart_json["common"] = root_path;
		restart_json["time"] = {{"t0", t0 + dt * t}};
		restart_json["output"] = {{"data", {{"file_index_offset", global_t}}}};

		restart_json["space"] = R"({
			"remesh": {
				"collapse": {
					"abs_max_edge_length": -1,
					"rel_max_edge_length": -1
				}
			}
		})"_json;

		const double starting_min_edge_length = stats.min_edge_length;
		restart_json["space"]["remesh"]["collapse"]["abs_max_edge_length"] = std::min(
			args["space"]["remesh"]["collapse"]["abs_max_edge_length"].get<double>(),
			starting_min_edge_length * args["space"]["remesh"]["collapse"]["rel_max_edge_length"].get<double>());
		restart_json["space"]["remesh"]["collapse"]["rel_max_edge_length"] = std::numeric_limits<float>::max();

		std::string rest_mesh_path = args["output"]["data"]["rest_mesh"].get<std::string>();
		if (!rest_mesh_path.empty())
		{
			if (!rest_mesh_written)
				logger().warn("Restart JSON for {} references a rest mesh that this formulation does not write.", name());

			rest_mesh_path = resolve_output_path(fmt::format(args["output"]["data"]["rest_mesh"], global_t));

			std::vector<json> patch;
			if (args["geometry"].is_array())
			{
				const std::vector<json> in_geometry = args["geometry"];
				for (int i = 0; i < in_geometry.size(); ++i)
				{
					if (!in_geometry[i]["is_obstacle"].get<bool>())
					{
						patch.push_back({
							{"op", "remove"},
							{"path", fmt::format("/geometry/{}", i)},
						});
					}
				}

				const int remaining_geometry = in_geometry.size() - patch.size();
				assert(remaining_geometry >= 0);

				patch.push_back({
					{"op", "add"},
					{"path", fmt::format("/geometry/{}", remaining_geometry > 0 ? "0" : "-")},
					{"value",
					 {
						 {"mesh", rest_mesh_path},
					 }},
				});
			}
			else
			{
				assert(args["geometry"].is_object());
				patch.push_back({
					{"op", "remove"},
					{"path", "/geometry"},
				});
				patch.push_back({
					{"op", "replace"},
					{"path", "/geometry"},
					{"value",
					 {
						 {"mesh", rest_mesh_path},
					 }},
				});
			}

			restart_json["patch"] = patch;
		}

		restart_json["input"] = {{
			"data",
			{
				{"state", resolve_output_path(fmt::format(args["output"]["data"]["state"], global_t))},
			},
		}};

		std::ofstream file(resolve_output_path(fmt::format(restart_json_path, global_t)));
		file << restart_json;
	}

	int VarForm::output_file_index(const int t) const
	{
		return t + args["output"]["data"]["file_index_offset"].get<int>();
	}

	std::string VarForm::resolve_input_path(const std::string &path, const bool only_if_exists) const
	{
		return utils::resolve_path(path, root_path, only_if_exists);
	}

	std::string VarForm::resolve_output_path(const std::string &path) const
	{
		if (output_path.empty() || path.empty() || std::filesystem::path(path).is_absolute())
		{
			return path;
		}
		return std::filesystem::weakly_canonical(std::filesystem::path(output_path) / path).string();
	}

	io::OutStatsData VarForm::compute_errors(const Eigen::MatrixXd &solution)
	{
		if (!args["output"]["advanced"]["compute_error"])
			return stats;

		double tend = 0;
		if (!args["time"].is_null())
			tend = args["time"]["tend"];

		stats.compute_errors(n_bases, bases, geom_bases(), *mesh_, *problem, tend, solution);
		return stats;
	}

	void VarForm::rebuild_node_positions(
		const std::vector<basis::ElementBases> &bases,
		const std::vector<int> &node_ids,
		std::vector<RowVectorNd> &positions)
	{
		positions.resize(node_ids.size());
		for (int n = 0; n < int(node_ids.size()); ++n)
		{
			const int node_id = node_ids[n];
			bool found = false;
			for (const auto &bs : bases)
			{
				for (const auto &b : bs.bases)
				{
					for (const auto &lg : b.global())
					{
						if (lg.index == node_id)
						{
							positions[n] = lg.node;
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
} // namespace polyfem::varform

#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/Evaluator.hpp>
#include <polyfem/varforms/VarFormUtils.hpp>

#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/refinement/APriori.hpp>
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
	VarForm::VarForm(const Units &units, const json &args, const std::string &out_path)
		: units(units), args(args), output_path(out_path)
	{
		if (utils::is_param_valid(args, "root_path"))
			root_path = args["root_path"].get<std::string>();
		else
			root_path = "";
	}

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
		const Eigen::VectorXi &disc_orders = legacy_primary_space_dont_use().disc_orders;
		const int discr_order = std::max(disc_orders.maxCoeff(), gdiscr_order);

		const int n_b_samples = std::max(n_b_samples_j, AssemblerUtils::quadrature_order("Mass", discr_order, AssemblerUtils::BasisType::POLY, mesh_->dimension()));
		return {{n_b_samples, n_b_samples}};
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
		rhs.resize(0, 0);
		build_fe_space(*mesh_, args);
		build_assembler_cache(*mesh_, args);
		build_boundary_condition(*mesh_, args);
		build_solution_layout();
		assemble_rhs(*mesh_, args);
		assemble_mass_mat(*mesh_, args);
		prepared_ = true;
	}

	void VarForm::solve(Eigen::MatrixXd &sol)
	{
		prepare();
		solve_problem(sol);
	}

	void VarForm::build_node_mapping(const mesh::Mesh &mesh, const json &args)
	{
		const FESpace &space = legacy_primary_space_dont_use();
		const Eigen::VectorXi &disc_orders = space.disc_orders;
		const std::shared_ptr<polyfem::mesh::MeshNodes> &mesh_nodes = space.mesh_nodes;

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
		const FESpace &space = legacy_primary_space_dont_use();
		const std::shared_ptr<GeometryMapping> &geometry = legacy_primary_geometry_dont_use();
		const int n_bases = space.n_bases;
		const Eigen::VectorXi &disc_orders = space.disc_orders;
		const Eigen::VectorXi &disc_ordersq = space.disc_ordersq;
		const bool isoparametric = geometry && (geometry->bases == space.bases);
		const int primary_size = n_bases * problem_dimension();
		const Eigen::MatrixXd stats_solution =
			solution.rows() >= primary_size
				? solution.topRows(primary_size).eval()
				: solution;

		nlohmann::json j;
		stats.save_json(
			args, n_bases, n_auxiliary_bases,
			stats_solution, *mesh_, disc_orders, disc_ordersq, *problem,
			timings, assembler ? assembler->name() : name(), isoparametric,
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

		const FESpace &space = legacy_primary_space_dont_use();
		const int n_bases = space.n_bases;
		const std::vector<basis::ElementBases> &bases = *space.bases;
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
		const std::shared_ptr<GeometryMapping> &geometry = legacy_primary_geometry_dont_use();
		if (!geometry || !geometry->mesh_nodes)
			log_and_throw_error("Primitive-to-node mapping is unavailable for this basis");

		auto indices = geometry->mesh_nodes->primitive_to_node();
		indices.resize(mesh_->n_vertices());
		return indices;
	}

	std::vector<int> VarForm::node_to_primitive() const
	{
		const std::shared_ptr<GeometryMapping> &geometry = legacy_primary_geometry_dont_use();
		auto p2n = primitive_to_node();
		std::vector<int> indices;
		indices.resize(geometry->n_bases);
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
		const FESpace &space = legacy_primary_space_dont_use();
		const AssemblyCaches &caches = legacy_primary_caches_dont_use();
		const int n_bases = space.n_bases;
		const std::vector<basis::ElementBases> &bases = *space.bases;
		const assembler::AssemblyValsCache &mass_ass_vals_cache = caches.mass;
		const assembler::AssemblyValsCache &pure_mass_ass_vals_cache = caches.pure_mass;

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
		const FESpace &space = legacy_primary_space_dont_use();
		const AssemblyCaches &caches = legacy_primary_caches_dont_use();
		const VarFormBoundaryState &boundary = boundary_state();
		const int n_bases = space.n_bases;
		const std::vector<basis::ElementBases> &bases = *space.bases;
		const assembler::AssemblyValsCache &mass_ass_vals_cache = caches.mass;
		const std::vector<int> &dirichlet_nodes = boundary.dirichlet_nodes;
		const std::vector<int> &neumann_nodes = boundary.neumann_nodes;
		const std::vector<RowVectorNd> &dirichlet_nodes_position = boundary.dirichlet_nodes_position;
		const std::vector<RowVectorNd> &neumann_nodes_position = boundary.neumann_nodes_position;

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
		const FESpace &fe_space = legacy_primary_space_dont_use();
		const int n_bases = fe_space.n_bases;
		const std::vector<basis::ElementBases> &bases = *fe_space.bases;
		const Eigen::VectorXi &disc_orders = fe_space.disc_orders;
		const Eigen::VectorXi &disc_ordersq = fe_space.disc_ordersq;

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
		const FESpace &space = legacy_primary_space_dont_use();
		const std::shared_ptr<GeometryMapping> &geometry = legacy_primary_geometry_dont_use();
		const VarFormBoundaryState &boundary = boundary_state();
		const Eigen::VectorXi &disc_orders = space.disc_orders;
		const Eigen::VectorXi &disc_ordersq = space.disc_ordersq;
		const std::map<int, Eigen::MatrixXd> &polys = geometry->polys;
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d = geometry->polys_3d;
		const std::vector<mesh::LocalBoundary> &total_local_boundary = boundary.total_local_boundary;
		const std::vector<int> &dirichlet_nodes = boundary.dirichlet_nodes;
		const std::vector<RowVectorNd> &dirichlet_nodes_position = boundary.dirichlet_nodes_position;

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

	void VarForm::save_step_state(
		const double t0,
		const double dt,
		const int t,
		const time_integrator::ImplicitTimeIntegrator *time_integrator) const
	{
		const int global_t = output_file_index(t);
		const std::string state_path = resolve_output_path(fmt::format(args["output"]["data"]["state"], global_t));
		if (!state_path.empty() && time_integrator)
			time_integrator->save_state(state_path);

		save_restart_json(t0, dt, t);
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

	void VarForm::save_restart_json(const double t0, const double dt, const int t) const
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

		const FESpace &space = legacy_primary_space_dont_use();
		const int n_bases = space.n_bases;
		const std::vector<basis::ElementBases> &bases = *space.bases;

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

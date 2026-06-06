#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/io/MatrixIO.hpp>

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
#include <polyfem/refinement/APriori.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/Logger.hpp>

#include <igl/Timer.h>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/StringUtils.hpp>

#include <fstream>
#include <limits>

#include <spdlog/fmt/fmt.h>

namespace polyfem::varform
{
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
	} // namespace

	void VarForm::reset()
	{
		stats.reset();
		output_sampler_initialized_ = false;
		forms.clear();
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
		solve_data = solver::SolveData();
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
		mesh_ = nullptr;
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

	void VarForm::set_mesh(std::unique_ptr<mesh::Mesh> mesh)
	{
		mesh_ = std::move(mesh);
		output_sampler_initialized_ = false;
		if (!mesh_)
			return;

		load_mesh(*mesh_, args);
	}

	void VarForm::prepare()
	{
		if (!mesh_)
		{
			logger().error("Load the mesh first!");
			return;
		}

		mesh_->prepare_mesh();
		build_basis(*mesh_, should_use_iso_parametric(*mesh_, args), args);
		assemble_rhs(*mesh_, args);
		assemble_mass_mat(*mesh_, args);
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

	void VarForm::solve(Eigen::MatrixXd &sol)
	{
		prepare();
		solve_problem(sol);
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

	io::OutStatsData VarForm::compute_errors(const Eigen::MatrixXd &solution)
	{
		return stats;
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
			return output_fields(sample, solution, io::OutputFieldOptions{fields});
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
		double tend = args.value("tend", 1.0);
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
			vis_mesh_path,
			is_contact_enabled());
	}

	int VarForm::problem_dimension() const
	{
		if (!problem)
			return 0;
		if (problem->is_scalar())
			return 1;
		return mesh_ ? mesh_->dimension() : 0;
	}

	void VarForm::save_step_state(const double t0, const double dt, const int t, const Eigen::MatrixXd &sol) const
	{
		save_restart_json(t0, dt, t);
	}

	void VarForm::save_timestep(const double time, const int t, const double t0, const double dt, const Eigen::MatrixXd &solution) const
	{
		const io::OutputSpace space = output_space();
		if (!space.mesh || !args["output"]["advanced"]["save_time_sequence"])
			return;
		if (t % args["output"]["paraview"]["skip_frame"].get<int>())
			return;

		ensure_output_sampler();

		logger().trace("Saving VTU...");
		const std::string step_name = args["output"]["advanced"]["timestep_prefix"];
		const auto opts = export_options(space);
		output_geometry_.save_vtu(
			resolve_output_path(fmt::format(step_name + "{:d}.vtu", t)),
			space, output_field_function(solution, opts), time, dt,
			opts,
			is_contact_enabled());

		output_geometry_.save_pvd(
			resolve_output_path(args["output"]["paraview"]["file_name"]),
			[step_name](int i) { return fmt::format(step_name + "{:d}.vtm", i); },
			t, t0, dt, args["output"]["paraview"]["skip_frame"].get<int>());
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
			opts,
			is_contact_enabled());
	}

	void VarForm::save_restart_json(const double t0, const double dt, const int t) const
	{
		const std::string restart_json_path = args["output"]["restart_json"];
		if (restart_json_path.empty())
			return;

		json restart_json;
		restart_json["root_path"] = root_path;
		restart_json["common"] = root_path;
		restart_json["time"] = {{"t0", t0 + dt * t}};

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
			rest_mesh_path = resolve_output_path(fmt::format(args["output"]["data"]["rest_mesh"], t));

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
				{"state", resolve_output_path(fmt::format(args["output"]["data"]["state"], t))},
			},
		}};

		std::ofstream file(resolve_output_path(fmt::format(restart_json_path, t)));
		file << restart_json;
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
} // namespace polyfem::varform

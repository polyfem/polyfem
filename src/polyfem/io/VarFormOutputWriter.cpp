#include "VarFormOutputWriter.hpp"

#include <polyfem/io/MshWriter.hpp>
#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/basis/Basis.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/solver/SolveData.hpp>
#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/Logger.hpp>

#include <filesystem>
#include <fstream>
#include <limits>

#include <spdlog/fmt/fmt.h>

namespace polyfem::io
{
	VarFormOutputWriter::VarFormOutputWriter(const varform::VarForm &var_form)
		: var_form_(var_form)
	{
	}

	std::string VarFormOutputWriter::resolve_output_path(const std::string &path) const
	{
		const OutputState out = var_form_.output_state();
		if (out.output_path.empty() || path.empty() || std::filesystem::path(path).is_absolute())
			return path;

		return std::filesystem::weakly_canonical(std::filesystem::path(out.output_path) / path).string();
	}

	void VarFormOutputWriter::ensure_sampler()
	{
		if (sampler_initialized_)
			return;

		const OutputState out = var_form_.output_state();
		if (out.mesh)
		{
			out_geom_.init_sampler(*out.mesh, out.args["output"]["paraview"]["vismesh_rel_area"]);
			out_geom_.build_grid(*out.mesh, out.args["output"]["advanced"]["sol_on_grid"]);
		}
		sampler_initialized_ = true;
	}

	OutGeometryData::ExportOptions VarFormOutputWriter::export_options(const OutputState &out) const
	{
		return OutGeometryData::ExportOptions(
			out.args,
			out.mesh->is_linear(),
			out.mesh->has_prism(),
			out.problem->is_scalar());
	}

	bool VarFormOutputWriter::is_contact_enabled(const OutputState &out) const
	{
		return out.args["contact"]["enabled"];
	}

	void VarFormOutputWriter::export_data(const Eigen::MatrixXd &sol, const Eigen::MatrixXd &pressure)
	{
		const OutputState out = var_form_.output_state();
		if (!out.mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (out.n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}
		if (sol.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		ensure_sampler();

		const std::string vis_mesh_path = resolve_output_path(out.args["output"]["paraview"]["file_name"]);
		const std::string nodes_path = resolve_output_path(out.args["output"]["data"]["nodes"]);
		const std::string solution_path = resolve_output_path(out.args["output"]["data"]["solution"]);
		const std::string stress_path = resolve_output_path(out.args["output"]["data"]["stress_mat"]);
		const std::string mises_path = resolve_output_path(out.args["output"]["data"]["mises"]);

		double tend = out.args.value("tend", 1.0);
		double dt = 1;
		if (!out.args["time"].is_null())
			dt = out.args["time"]["dt"];

		out_geom_.export_data(
			out, sol, pressure,
			!out.args["time"].is_null(),
			tend, dt,
			export_options(out),
			vis_mesh_path,
			nodes_path,
			solution_path,
			stress_path,
			mises_path,
			is_contact_enabled(out));
	}

	void VarFormOutputWriter::save_timestep(const double time, const int t, const double t0, const double dt, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &pressure)
	{
		const OutputState out = var_form_.output_state();
		if (!out.mesh || !out.args["output"]["advanced"]["save_time_sequence"])
			return;
		if (t % out.args["output"]["paraview"]["skip_frame"].get<int>())
			return;

		ensure_sampler();

		logger().trace("Saving VTU...");
		const std::string step_name = out.args["output"]["advanced"]["timestep_prefix"];
		out_geom_.save_vtu(
			resolve_output_path(fmt::format(step_name + "{:d}.vtu", t)),
			out, sol, pressure, time, dt,
			export_options(out),
			is_contact_enabled(out));

		out_geom_.save_pvd(
			resolve_output_path(out.args["output"]["paraview"]["file_name"]),
			[step_name](int i) { return fmt::format(step_name + "{:d}.vtm", i); },
			t, t0, dt, out.args["output"]["paraview"]["skip_frame"].get<int>());
	}

	void VarFormOutputWriter::save_subsolve(const int i, const int t, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &pressure)
	{
		const OutputState out = var_form_.output_state();
		if (!out.mesh || !out.args["output"]["advanced"]["save_solve_sequence_debug"].get<bool>())
			return;

		double dt = 1;
		if (!out.args["time"].is_null())
			dt = out.args["time"]["dt"];

		ensure_sampler();
		out_geom_.save_vtu(
			resolve_output_path(fmt::format("solve_{:d}.vtu", i)),
			out, sol, pressure, t, dt,
			export_options(out),
			is_contact_enabled(out));
	}

	void VarFormOutputWriter::save_json(const Eigen::MatrixXd &sol)
	{
		const OutputState output = var_form_.output_state();
		const std::string out_path = resolve_output_path(output.args["output"]["json"]);
		if (out_path.empty())
			return;

		std::ofstream file(out_path);
		if (!file.is_open())
		{
			logger().error("Unable to save simulation JSON to {}", out_path);
			return;
		}
		save_json(sol, file);
	}

	void VarFormOutputWriter::save_json(const Eigen::MatrixXd &sol, std::ostream &out)
	{
		const OutputState output = var_form_.output_state();
		if (!output.mesh)
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
		nlohmann::json j;
		output.stats.save_json(
			output.args, output.n_bases, output.n_pressure_bases,
			sol, *output.mesh, output.disc_orders, output.disc_ordersq, *output.problem,
			output.timings, output.formulation, output.iso_parametric,
			output.args["output"]["advanced"]["sol_at_node"], j);
		out << j.dump(4) << std::endl;
	}

	void VarFormOutputWriter::build_mesh_matrices(const OutputState &out, Eigen::MatrixXd &V, Eigen::MatrixXi &F) const
	{
		assert(out.mesh);
		assert(out.bases.size() == out.mesh->n_elements());
		const size_t n_vertices = out.n_bases - out.obstacle.n_vertices();
		const int dim = out.mesh->dimension();

		V.resize(n_vertices, dim);
		F.resize(out.bases.size(), dim + 1);

		for (int i = 0; i < out.bases.size(); i++)
		{
			const basis::ElementBases &element = out.bases[i];
			for (int j = 0; j < element.bases.size(); j++)
			{
				const basis::Basis &basis = element.bases[j];
				assert(basis.global().size() == 1);
				V.row(basis.global()[0].index) = basis.global()[0].node;
				if (j < F.cols())
					F(i, j) = basis.global()[0].index;
			}
		}
	}

	void VarFormOutputWriter::save_step_state(const double t0, const double dt, const int t, const Eigen::MatrixXd &sol)
	{
		const OutputState out = var_form_.output_state();

		const std::string rest_mesh_path = out.args["output"]["data"]["rest_mesh"].get<std::string>();
		if (!rest_mesh_path.empty())
		{
			Eigen::MatrixXd V;
			Eigen::MatrixXi F;
			build_mesh_matrices(out, V, F);
			io::MshWriter::write(
				resolve_output_path(fmt::format(out.args["output"]["data"]["rest_mesh"], t)),
				V, F, out.mesh->get_body_ids(), out.mesh->is_volume(), /*binary=*/true);
		}

		const std::string state_path = resolve_output_path(fmt::format(out.args["output"]["data"]["state"], t));
		if (!state_path.empty() && out.solve_data.time_integrator)
			out.solve_data.time_integrator->save_state(state_path);

		save_restart_json(t0, dt, t);
	}

	void VarFormOutputWriter::save_restart_json(const double t0, const double dt, const int t) const
	{
		const OutputState out = var_form_.output_state();
		const std::string restart_json_path = out.args["output"]["restart_json"];
		if (restart_json_path.empty())
			return;

		json restart_json;
		restart_json["root_path"] = out.root_path;
		restart_json["common"] = out.root_path;
		restart_json["time"] = {{"t0", t0 + dt * t}};

		restart_json["space"] = R"({
			"remesh": {
				"collapse": {
					"abs_max_edge_length": -1,
					"rel_max_edge_length": -1
				}
			}
		})"_json;

		const double starting_min_edge_length =
			out.starting_min_edge_length > 0 ? out.starting_min_edge_length : out.stats.min_edge_length;
		restart_json["space"]["remesh"]["collapse"]["abs_max_edge_length"] = std::min(
			out.args["space"]["remesh"]["collapse"]["abs_max_edge_length"].get<double>(),
			starting_min_edge_length * out.args["space"]["remesh"]["collapse"]["rel_max_edge_length"].get<double>());
		restart_json["space"]["remesh"]["collapse"]["rel_max_edge_length"] = std::numeric_limits<float>::max();

		std::string rest_mesh_path = out.args["output"]["data"]["rest_mesh"].get<std::string>();
		if (!rest_mesh_path.empty())
		{
			rest_mesh_path = resolve_output_path(fmt::format(out.args["output"]["data"]["rest_mesh"], t));

			std::vector<json> patch;
			if (out.args["geometry"].is_array())
			{
				const std::vector<json> in_geometry = out.args["geometry"];
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
				assert(out.args["geometry"].is_object());
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
				{"state", resolve_output_path(fmt::format(out.args["output"]["data"]["state"], t))},
			},
		}};

		std::ofstream file(resolve_output_path(fmt::format(restart_json_path, t)));
		file << restart_json;
	}
} // namespace polyfem::io

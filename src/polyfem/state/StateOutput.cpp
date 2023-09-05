#include <polyfem/State.hpp>

#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Timer.hpp>

#include <filesystem>

namespace polyfem
{
	void State::compute_errors(const Eigen::MatrixXd &sol)
	{
		if (!args["output"]["advanced"]["compute_error"])
			return;

		double tend = 0;

		if (!args["time"].is_null())
		{
			tend = args["time"]["tend"];
		}

		stats.compute_errors(n_bases, bases, geom_bases(), *mesh, *problem, tend, sol);
	}

	std::string State::root_path() const
	{
		if (utils::is_param_valid(args, "root_path"))
			return args["root_path"].get<std::string>();
		return "";
	}

	std::string State::resolve_input_path(const std::string &path, const bool only_if_exists) const
	{
		return utils::resolve_path(path, root_path(), only_if_exists);
	}

	std::string State::resolve_output_path(const std::string &path) const
	{
		if (output_dir.empty() || path.empty() || std::filesystem::path(path).is_absolute())
		{
			return path;
		}
		return std::filesystem::weakly_canonical(std::filesystem::path(output_dir) / path).string();
	}

	void State::save_timestep(const double time, const int t, const double t0, const double dt, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &pressure)
	{
		if (args["output"]["advanced"]["save_time_sequence"] && !(t % args["output"]["paraview"]["skip_frame"].get<int>()))
		{
			logger().trace("Saving VTU...");
			POLYFEM_SCOPED_TIMER("Saving VTU");
			const std::string step_name = args["output"]["advanced"]["timestep_prefix"];

			if (!solve_export_to_file)
				solution_frames.emplace_back();

			out_geom.save_vtu(
				resolve_output_path(fmt::format(step_name + "{:d}.vtu", t)),
				*this, sol, pressure, time, dt,
				io::OutGeometryData::ExportOptions(args, mesh->is_linear(), problem->is_scalar(), solve_export_to_file),
				is_contact_enabled(), solution_frames);

			out_geom.save_pvd(
				resolve_output_path(args["output"]["paraview"]["file_name"]),
				[step_name](int i) { return fmt::format(step_name + "{:d}.vtm", i); },
				t, t0, dt, args["output"]["paraview"]["skip_frame"].get<int>());
		}
	}

	void State::save_json(const Eigen::MatrixXd &sol)
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
			save_json(sol, out);
			out.close();
		}
	}

	void State::save_json(const Eigen::MatrixXd &sol, std::ostream &out)
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

		using json = nlohmann::json;
		json j;
		stats.save_json(args, n_bases, n_pressure_bases,
						sol, *mesh, disc_orders, *problem, timings,
						assembler->name(), iso_parametric(), args["output"]["advanced"]["sol_at_node"],
						j);
		out << j.dump(4) << std::endl;
	}

	void State::save_subsolve(const int i, const int t, const Eigen::MatrixXd &sol, const Eigen::MatrixXd &pressure)
	{
		if (!args["output"]["advanced"]["save_solve_sequence_debug"].get<bool>())
			return;

		if (!solve_export_to_file)
			solution_frames.emplace_back();

		double dt = 1;
		if (!args["time"].is_null())
			dt = args["time"]["dt"];

		out_geom.save_vtu(
			resolve_output_path(fmt::format("solve_{:d}.vtu", i)),
			*this, sol, pressure, t, dt,
			io::OutGeometryData::ExportOptions(args, mesh->is_linear(), problem->is_scalar(), solve_export_to_file),
			is_contact_enabled(), solution_frames);
	}

	void State::export_data(const Eigen::MatrixXd &sol, const Eigen::MatrixXd &pressure)
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

		// Export vtu mesh of solution + wire mesh of deformed input
		// + mesh colored with the bases
		const std::string vis_mesh_path = resolve_output_path(args["output"]["paraview"]["file_name"]);
		const std::string nodes_path = resolve_output_path(args["output"]["data"]["nodes"]);
		const std::string solution_path = resolve_output_path(args["output"]["data"]["solution"]);
		const std::string stress_path = resolve_output_path(args["output"]["data"]["stress_mat"]);
		const std::string mises_path = resolve_output_path(args["output"]["data"]["mises"]);

		double tend = args.value("tend", 1.0);
		double dt = 1;
		if (!args["time"].is_null())
			dt = args["time"]["dt"];

		out_geom.export_data(
			*this, sol, pressure,
			!args["time"].is_null(),
			tend, dt,
			io::OutGeometryData::ExportOptions(args, mesh->is_linear(), problem->is_scalar(), solve_export_to_file),
			vis_mesh_path,
			nodes_path,
			solution_path,
			stress_path,
			mises_path,
			is_contact_enabled(), solution_frames);
	}

	void State::save_restart_json(const double t0, const double dt, const int t) const
	{
		const std::string restart_json_path = args["output"]["restart_json"];
		if (restart_json_path.empty())
			return;

		json restart_json;
		restart_json["root_path"] = root_path();
		restart_json["common"] = root_path();
		restart_json["time"] = {{"t0", t0 + dt * t}};

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
						 // TODO: this does not set the surface selections
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
						 // TODO: this does not set the surface selections
						 {"mesh", rest_mesh_path},
					 }},
				});
			}

			restart_json["patch"] = patch;
		}

		restart_json["input"] = {{
			"data",
			{
				{"u_path", resolve_output_path(fmt::format(args["output"]["data"]["u_path"], t))},
				{"v_path", resolve_output_path(fmt::format(args["output"]["data"]["v_path"], t))},
				{"a_path", resolve_output_path(fmt::format(args["output"]["data"]["a_path"], t))},
			},
		}};

		std::ofstream file(resolve_output_path(fmt::format(restart_json_path, t)));
		file << restart_json;
	}
} // namespace polyfem

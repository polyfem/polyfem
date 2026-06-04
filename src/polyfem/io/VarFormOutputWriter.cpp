#include "VarFormOutputWriter.hpp"

#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/utils/Logger.hpp>

#include <spdlog/fmt/fmt.h>

namespace polyfem::io
{
	VarFormOutputWriter::VarFormOutputWriter(const varform::VarForm &var_form)
		: var_form_(var_form)
	{
	}

	std::string VarFormOutputWriter::resolve_output_path(const std::string &path) const
	{
		return var_form_.resolve_output_path(path);
	}

	void VarFormOutputWriter::ensure_sampler()
	{
		if (sampler_initialized_)
			return;

		const OutputSpace space = var_form_.output_space();
		if (space.mesh)
		{
			out_geom_.init_sampler(*space.mesh, var_form_.input_args()["output"]["paraview"]["vismesh_rel_area"]);
			out_geom_.build_grid(*space.mesh, var_form_.input_args()["output"]["advanced"]["sol_on_grid"]);
		}
		sampler_initialized_ = true;
	}

	OutGeometryData::ExportOptions VarFormOutputWriter::export_options(const OutputSpace &space) const
	{
		return OutGeometryData::ExportOptions(
			var_form_.input_args(),
			space.mesh->is_linear(),
			space.mesh->has_prism(),
			var_form_.problem_dimension() == 1);
	}

	OutputFieldFunction VarFormOutputWriter::output_field_function(const Eigen::MatrixXd &sol, const OutGeometryData::ExportOptions &opts) const
	{
		return [this, &sol, fields = opts.fields](const OutputSample &sample) {
			return var_form_.output_fields(sample, sol, OutputFieldOptions{fields});
		};
	}

	void VarFormOutputWriter::export_data(const Eigen::MatrixXd &sol)
	{
		const OutputSpace space = var_form_.output_space();
		if (!space.mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		if (sol.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		ensure_sampler();

		const json &args = var_form_.input_args();
		const std::string vis_mesh_path = resolve_output_path(args["output"]["paraview"]["file_name"]);
		const bool has_time = args.contains("time") && !args["time"].is_null();
		double tend = args.value("tend", 1.0);
		double dt = 1;
		if (has_time)
			dt = args["time"]["dt"];

		const auto opts = export_options(space);
		out_geom_.export_data(
			space,
			output_field_function(sol, opts),
			has_time,
			tend, dt,
			opts,
			vis_mesh_path,
			var_form_.is_contact_enabled());
	}

	void VarFormOutputWriter::save_timestep(const double time, const int t, const double t0, const double dt, const Eigen::MatrixXd &sol)
	{
		const OutputSpace space = var_form_.output_space();
		const json &args = var_form_.input_args();
		if (!space.mesh || !args["output"]["advanced"]["save_time_sequence"])
			return;
		if (t % args["output"]["paraview"]["skip_frame"].get<int>())
			return;

		ensure_sampler();

		logger().trace("Saving VTU...");
		const std::string step_name = args["output"]["advanced"]["timestep_prefix"];
		const auto opts = export_options(space);
		out_geom_.save_vtu(
			resolve_output_path(fmt::format(step_name + "{:d}.vtu", t)),
			space, output_field_function(sol, opts), time, dt,
			opts,
			var_form_.is_contact_enabled());

		out_geom_.save_pvd(
			resolve_output_path(args["output"]["paraview"]["file_name"]),
			[step_name](int i) { return fmt::format(step_name + "{:d}.vtm", i); },
			t, t0, dt, args["output"]["paraview"]["skip_frame"].get<int>());
	}

	void VarFormOutputWriter::save_subsolve(const int i, const int t, const Eigen::MatrixXd &sol)
	{
		const OutputSpace space = var_form_.output_space();
		const json &args = var_form_.input_args();
		if (!space.mesh || !args["output"]["advanced"]["save_solve_sequence_debug"].get<bool>())
			return;

		const bool has_time = args.contains("time") && !args["time"].is_null();
		double dt = 1;
		if (has_time)
			dt = args["time"]["dt"];

		ensure_sampler();
		const auto opts = export_options(space);
		out_geom_.save_vtu(
			resolve_output_path(fmt::format("solve_{:d}.vtu", i)),
			space, output_field_function(sol, opts), t, dt,
			opts,
			var_form_.is_contact_enabled());
	}

} // namespace polyfem::io

#include <polyfem/varforms/VarFormFactory.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/varforms/NonlinearElasticTransientVarForm.hpp>

namespace polyfem::varform
{
	namespace
	{
		bool is_enabled(const json &args, const json::json_pointer &ptr)
		{
			return args.contains(ptr) && args.at(ptr).is_boolean() && args.at(ptr).get<bool>();
		}

		bool has_nonempty_string(const json &args, const json::json_pointer &ptr)
		{
			return args.contains(ptr) && args.at(ptr).is_string() && !args.at(ptr).get<std::string>().empty();
		}
	} // namespace

	bool VarFormFactory::supports(const std::string &formulation, const json &args)
	{
		// Keep the migration opt-in and boring: only transient nonlinear tensor
		// elasticity is routed through the new VarForm path for now.
		if (!args.contains("time") || args["time"].is_null())
			return false;

		// Output/restart still lives in State. Until VarForm owns that surface,
		// route output-producing solves through the legacy path.
		if (is_enabled(args, "/output/advanced/save_time_sequence"_json_pointer)
			|| is_enabled(args, "/output/advanced/save_solve_sequence_debug"_json_pointer)
			|| is_enabled(args, "/output/advanced/save_ccd_debug_meshes"_json_pointer)
			|| is_enabled(args, "/output/advanced/save_nl_solve_sequence"_json_pointer)
			|| is_enabled(args, "/output/stats"_json_pointer)
			|| has_nonempty_string(args, "/output/paraview/file_name"_json_pointer)
			|| has_nonempty_string(args, "/output/json"_json_pointer)
			|| has_nonempty_string(args, "/output/restart_json"_json_pointer)
			|| has_nonempty_string(args, "/output/data/solution"_json_pointer)
			|| has_nonempty_string(args, "/output/data/state"_json_pointer)
			|| has_nonempty_string(args, "/output/data/rest_mesh"_json_pointer)
			|| has_nonempty_string(args, "/output/data/mises"_json_pointer)
			|| has_nonempty_string(args, "/output/data/nodes"_json_pointer)
			|| has_nonempty_string(args, "/input/data/state"_json_pointer))
		{
			return false;
		}

		if (args.value("/space/remesh/enabled"_json_pointer, false))
			return false;

		if (args.value("/contact/periodic"_json_pointer, false))
			return false;

		if (args.contains("/boundary_conditions/periodic_boundary/linear_displacement_offset"_json_pointer)
			&& args.at("/boundary_conditions/periodic_boundary/linear_displacement_offset"_json_pointer).size() > 0)
			return false;

		const auto assembler = assembler::AssemblerUtils::make_assembler(formulation);
		if (!assembler || assembler->is_linear() || !assembler->is_tensor() || assembler->is_fluid())
			return false;

		if (!assembler::AssemblerUtils::other_assembler_name(formulation).empty())
			return false;

		return true;
	}

	std::shared_ptr<VarForm> VarFormFactory::create(const std::string &formulation, const json &args)
	{
		if (!supports(formulation, args))
			return nullptr;

		return std::make_shared<NonlinearElasticTransientVarForm>();
	}
} // namespace polyfem::varform

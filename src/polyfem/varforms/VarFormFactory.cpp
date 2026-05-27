#include <polyfem/varforms/VarFormFactory.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/varforms/NonlinearElasticTransientVarForm.hpp>

namespace polyfem::varform
{
	bool VarFormFactory::supports(const std::string &formulation, const json &args)
	{
		if (args.value("/space/remesh/enabled"_json_pointer, false))
			return false;

		if (args.value("/contact/periodic"_json_pointer, false))
			return false;

		if (args.contains("/boundary_conditions/periodic_boundary/linear_displacement_offset"_json_pointer)
			&& args.at("/boundary_conditions/periodic_boundary/linear_displacement_offset"_json_pointer).size() > 0)
			return false;

		const auto assembler = assembler::AssemblerUtils::make_assembler(formulation);
		if (!assembler || !assembler->is_tensor() || assembler->is_fluid())
			return false;

		if (!assembler::AssemblerUtils::other_assembler_name(formulation).empty())
			return false;

		if (!assembler->is_linear())
			return true;

		const bool has_contact = args.value("/contact/enabled"_json_pointer, false);
		const bool has_pressure =
			args["boundary_conditions"]["pressure_boundary"].size() > 0
			|| args["boundary_conditions"]["pressure_cavity"].size() > 0;
		const bool has_constraints =
			args.contains("constraints")
			&& (!args["constraints"]["hard"].empty() || !args["constraints"]["soft"].empty());

		return has_contact || has_pressure || has_constraints;
	}

	std::shared_ptr<VarForm> VarFormFactory::create(const std::string &formulation, const json &args)
	{
		if (!supports(formulation, args))
			return nullptr;

		if (args.contains("time") && !args["time"].is_null())
			return std::make_shared<NonlinearElasticTransientVarForm>();

		return std::make_shared<NonlinearElasticStaticVarForm>();
	}
} // namespace polyfem::varform

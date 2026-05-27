#include <polyfem/varforms/VarFormFactory.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/varforms/NonlinearElasticTransientVarForm.hpp>

namespace polyfem::varform
{
	bool VarFormFactory::supports(const std::string &formulation, const json &args)
	{
		// Keep the migration opt-in and boring: only transient nonlinear tensor
		// elasticity is routed through the new VarForm path for now.
		if (!args.contains("time") || args["time"].is_null())
			return false;

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

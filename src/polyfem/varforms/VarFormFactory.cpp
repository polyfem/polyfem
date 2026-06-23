#include <polyfem/varforms/VarFormFactory.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/varforms/BilaplacianVarForm.hpp>
#include <polyfem/varforms/FluidVarForm.hpp>
#include <polyfem/varforms/IncompressibleElasticVarForm.hpp>
#include <polyfem/varforms/LinearElasticVarForm.hpp>
#include <polyfem/varforms/NonlinearElasticVarForm.hpp>
#include <polyfem/varforms/ScalarVarForm.hpp>

namespace polyfem::varform
{
	namespace
	{
		bool has_entries(const json &args, const json::json_pointer &path)
		{
			return args.contains(path) && !args.at(path).empty();
		}
	} // namespace

	std::string formulation_from_args(const json &args)
	{
		if (!args.contains("materials") || args["materials"].is_null())
			return "";

		if (args["materials"].is_array())
		{
			std::string current;
			for (const auto &m : args["materials"])
			{
				const std::string tmp = m["type"];
				if (current.empty())
					current = tmp;
				else if (current != tmp)
				{
					if (assembler::AssemblerUtils::is_elastic_material(current)
						&& assembler::AssemblerUtils::is_elastic_material(tmp))
					{
						current = "MultiModels";
					}
					else
					{
						return "";
					}
				}
			}

			return current;
		}

		return args["materials"].value("type", "");
	}

	bool uses_varform_state(json args)
	{
		utils::apply_common_params(args);
		const std::string formulation = formulation_from_args(args);
		return !formulation.empty() && VarFormFactory::create(formulation, args) != nullptr;
	}

	bool VarFormFactory::supports(const std::string &formulation, const json &args)
	{
		if (args.value("/space/remesh/enabled"_json_pointer, false))
			return false;

		if (args.value("/contact/periodic"_json_pointer, false))
			return false;

		if (args.value("/boundary_conditions/periodic_boundary/enabled"_json_pointer, false))
			return false;

		if (args.contains("/boundary_conditions/periodic_boundary/linear_displacement_offset"_json_pointer)
			&& args.at("/boundary_conditions/periodic_boundary/linear_displacement_offset"_json_pointer).size() > 0)
			return false;

		const auto assembler = assembler::AssemblerUtils::make_assembler(formulation);
		if (!assembler)
			return false;

		if (formulation == "OperatorSplitting")
			return false;

		if (!assembler::AssemblerUtils::other_assembler_name(formulation).empty())
			return formulation == "Stokes"
				   || formulation == "NavierStokes"
				   || formulation == "IncompressibleLinearElasticity"
				   || formulation == "Bilaplacian";

		if (assembler->is_fluid())
			return false;

		return assembler->is_tensor() || assembler->is_linear();
	}

	std::shared_ptr<VarForm> VarFormFactory::create(const std::string &formulation, const json &args)
	{
		if (!supports(formulation, args))
			return nullptr;

		const auto assembler = assembler::AssemblerUtils::make_assembler(formulation);
		const bool has_contact = args.value("/contact/enabled"_json_pointer, false);
		const bool has_pressure = has_entries(args, "/boundary_conditions/pressure_boundary"_json_pointer)
								  || has_entries(args, "/boundary_conditions/pressure_cavity"_json_pointer);
		const bool has_constraints =
			has_entries(args, "/constraints/hard"_json_pointer)
			|| has_entries(args, "/constraints/soft"_json_pointer);

		if (formulation == "Stokes")
			return (!has_contact && !has_constraints) ? std::make_shared<StokesVarForm>() : nullptr;
		if (formulation == "NavierStokes")
			return (!has_contact && !has_constraints) ? std::make_shared<NavierStokesVarForm>() : nullptr;
		if (formulation == "IncompressibleLinearElasticity")
			return (!has_contact && !has_pressure && !has_constraints) ? std::make_shared<IncompressibleElasticVarForm>() : nullptr;
		if (formulation == "Bilaplacian")
			return (!has_contact && !has_constraints) ? std::make_shared<BilaplacianVarForm>() : nullptr;

		if (!assembler->is_tensor())
			return (!has_contact && !has_pressure && !has_constraints) ? std::make_shared<ScalarVarForm>() : nullptr;

		if (assembler->is_linear() && !has_contact && !has_pressure && !has_constraints)
			return std::make_shared<LinearElasticVarForm>();

		if (args.contains("time") && !args["time"].is_null())
			return std::make_shared<NonlinearElasticTransientVarForm>();

		return std::make_shared<NonlinearElasticStaticVarForm>();
	}
} // namespace polyfem::varform

#include <polyfem/varforms/VarFormFactory.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>
#include <polyfem/varforms/BilaplacianVarForm.hpp>
#include <polyfem/varforms/FluidVarForm.hpp>
#include <polyfem/varforms/IncompressibleElasticVarForm.hpp>
#include <polyfem/varforms/LinearElasticVarForm.hpp>
#include <polyfem/varforms/NonlinearElasticVarForm.hpp>
#include <polyfem/varforms/NavierStokesFSIVarForm.hpp>
#include <polyfem/varforms/OperatorSplittingVarForm.hpp>
#include <polyfem/varforms/ScalarVarForm.hpp>
#include <polyfem/varforms/ThermoElasticVarForm.hpp>
#include <polyfem/varforms/diff/DifferentiableLinearElasticVarForm.hpp>
#include <polyfem/varforms/diff/DifferentiableNonlinearElasticVarForm.hpp>
#include <polyfem/varforms/diff/DifferentiableScalarVarForm.hpp>

namespace polyfem::varform
{
	namespace
	{
		bool is_scalar_formulation(const std::string &formulation)
		{
			return formulation == "Helmholtz"
				   || formulation == "Laplacian"
				   || formulation == "Electrostatics";
		}

		bool is_linear_elastic_formulation(const std::string &formulation)
		{
			return formulation == "LinearElasticity"
				   || formulation == "HookeLinearElasticity";
		}

		bool is_nonlinear_elastic_formulation(const std::string &formulation)
		{
			return formulation == "SaintVenant"
				   || formulation == "NeoHookean"
				   || formulation == "IsochoricNeoHookean"
				   || formulation == "MooneyRivlin"
				   || formulation == "MooneyRivlin3Param"
				   || formulation == "MooneyRivlin3ParamSymbolic"
				   || formulation == "MultiModels"
				   || formulation == "MaterialSum"
				   || formulation == "UnconstrainedOgden"
				   || formulation == "IncompressibleOgden"
				   || formulation == "VolumePenalty"
				   || formulation == "InversionBarrier"
				   || formulation == "HGOFiber"
				   || formulation == "HGODispersion"
				   || formulation == "ActiveFiber"
				   || formulation == "AMIPS"
				   || formulation == "FixedCorotational";
		}

		bool is_elastic_formulation(const std::string &formulation)
		{
			return is_linear_elastic_formulation(formulation)
				   || is_nonlinear_elastic_formulation(formulation);
		}

		bool has_non_empty_entries(const json &args, const json::json_pointer &path)
		{
			return args.contains(path) && !args.at(path).empty();
		}

		bool has_two_mesh_fsi_material(const json &args)
		{
			if (!args.contains("materials") || args["materials"].is_null())
				return false;

			const auto has_solid_fields = [](const json &material) {
				return material.contains("fluid_geometry_id")
					   && material.contains("solid_geometry_id")
					   && material.contains("displacement_space_id")
					   && material.contains("solid_material");
			};

			const json &materials = args["materials"];
			if (!materials.is_array())
				return has_solid_fields(materials);
			if (materials.empty())
				return false;

			for (const json &material : materials)
				if (!has_solid_fields(material))
					return false;
			return true;
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
		return !formulation.empty() && VarFormFactory::supports(formulation, args);
	}

	bool VarFormFactory::supports(
		const std::string &formulation,
		const json &args,
		const bool is_optimization)
	{
		if (args.value("/space/remesh/enabled"_json_pointer, false))
			return false;

		const bool homogenization = args.contains("/constraints/macro_displacement_gradient"_json_pointer);
		const bool has_contact = args.value("/contact/enabled"_json_pointer, false);
		const bool has_periodic_constraints =
			has_non_empty_entries(args, "/boundary_conditions/periodic"_json_pointer);
		const bool periodic_contact = args.value("/contact/periodic"_json_pointer, false);
		if (periodic_contact)
		{
			if (!homogenization)
				return false;
			if (!has_contact)
				return false;
			if (!has_periodic_constraints)
				return false;
		}

		if (homogenization && !is_optimization)
			return false;
		if (homogenization && args.contains("time") && !args["time"].is_null())
			return false;

		const bool has_pressure = has_non_empty_entries(args, "/boundary_conditions/pressure_boundary"_json_pointer)
								  || has_non_empty_entries(args, "/boundary_conditions/pressure_cavity"_json_pointer);
		const bool has_file_constraints =
			has_non_empty_entries(args, "/constraints/hard"_json_pointer)
			|| has_non_empty_entries(args, "/constraints/soft"_json_pointer);
		const json zero_mean = args.contains("/constraints/zero_mean"_json_pointer)
								   ? args.at("/constraints/zero_mean"_json_pointer)
								   : json(false);
		const bool has_zero_mean_constraints =
			(zero_mean.is_boolean() && zero_mean.get<bool>())
			|| (zero_mean.is_array() && !zero_mean.empty());
		const bool has_constraints =
			has_file_constraints || has_periodic_constraints || has_zero_mean_constraints;

		if (formulation == "ThermoElasticity")
			return !is_optimization && !has_pressure && !has_constraints;

		if (formulation == "Stokes")
			return !is_optimization && !has_contact && !has_constraints;
		if (formulation == "NavierStokes")
			return !is_optimization && !has_contact && !has_constraints;
		if (formulation == "NavierStokesFSI")
			return !is_optimization
				   && args.contains("time") && !args["time"].is_null()
				   && (!has_contact || has_two_mesh_fsi_material(args)) && !has_constraints;
		if (formulation == "OperatorSplitting")
			return !is_optimization
				   && args.contains("time") && !args["time"].is_null()
				   && !has_contact && !has_constraints;
		if (formulation == "IncompressibleLinearElasticity")
			return !is_optimization && !has_contact && !has_pressure && !has_constraints;
		if (formulation == "Bilaplacian")
			return !is_optimization && !has_contact && !has_constraints;

		if (is_scalar_formulation(formulation))
		{
			return !homogenization && !has_contact && !has_pressure && !has_file_constraints;
		}

		return is_elastic_formulation(formulation);
	}

	std::shared_ptr<VarForm> VarFormFactory::create(
		const std::string &formulation,
		const json &args,
		const bool is_optimization)
	{
		if (!supports(formulation, args, is_optimization))
			return nullptr;

		if (formulation == "ThermoElasticity")
			return std::make_shared<ThermoElasticVarForm>();
		if (formulation == "Stokes")
			return std::make_shared<StokesVarForm>();
		if (formulation == "NavierStokes")
			return std::make_shared<NavierStokesVarForm>();
		if (formulation == "NavierStokesFSI")
			return std::make_shared<NavierStokesFSIVarForm>();
		if (formulation == "OperatorSplitting")
			return std::make_shared<OperatorSplittingVarForm>();
		if (formulation == "IncompressibleLinearElasticity")
			return std::make_shared<IncompressibleElasticVarForm>();
		if (formulation == "Bilaplacian")
			return std::make_shared<BilaplacianVarForm>();

		if (is_scalar_formulation(formulation))
			return is_optimization
					   ? std::static_pointer_cast<VarForm>(std::make_shared<DifferentiableScalarVarForm>())
					   : std::make_shared<ScalarVarForm>();

		const bool homogenization = args.contains("/constraints/macro_displacement_gradient"_json_pointer);

		if (homogenization)
			return std::make_shared<DifferentiableNonlinearElasticStaticVarForm>();

		const bool has_contact = args.value("/contact/enabled"_json_pointer, false);
		const bool has_pressure = has_non_empty_entries(args, "/boundary_conditions/pressure_boundary"_json_pointer)
								  || has_non_empty_entries(args, "/boundary_conditions/pressure_cavity"_json_pointer);
		const bool has_file_constraints =
			has_non_empty_entries(args, "/constraints/hard"_json_pointer)
			|| has_non_empty_entries(args, "/constraints/soft"_json_pointer);
		const bool has_periodic_constraints =
			has_non_empty_entries(args, "/boundary_conditions/periodic"_json_pointer);
		const json zero_mean = args.contains("/constraints/zero_mean"_json_pointer)
								   ? args.at("/constraints/zero_mean"_json_pointer)
								   : json(false);
		const bool has_zero_mean_constraints =
			(zero_mean.is_boolean() && zero_mean.get<bool>())
			|| (zero_mean.is_array() && !zero_mean.empty());
		const bool has_constraints =
			has_file_constraints || has_periodic_constraints || has_zero_mean_constraints;

		if (is_linear_elastic_formulation(formulation)
			&& !has_contact && !has_pressure && !has_constraints)
		{
			return is_optimization
					   ? std::static_pointer_cast<VarForm>(std::make_shared<DifferentiableLinearElasticVarForm>())
					   : std::make_shared<LinearElasticVarForm>();
		}

		if (args.contains("time") && !args["time"].is_null())
		{
			return is_optimization
					   ? std::static_pointer_cast<VarForm>(std::make_shared<DifferentiableNonlinearElasticTransientVarForm>())
					   : std::make_shared<NonlinearElasticTransientVarForm>();
		}

		return is_optimization
				   ? std::static_pointer_cast<VarForm>(std::make_shared<DifferentiableNonlinearElasticStaticVarForm>())
				   : std::make_shared<NonlinearElasticStaticVarForm>();
	}
} // namespace polyfem::varform

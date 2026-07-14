#include <polyfem/materials/LameParameter.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/CudaBoth.hpp>

#include <string>
#include <stdexcept>

namespace polyfem::material
{
	namespace
	{
		utils::ExpressionValue parse_expr(const json &value, const std::string &unit_type, const std::string &root_path)
		{
			utils::ExpressionValue out;
			out.init(value, root_path);
			out.set_unit_type(unit_type);
			return out;
		}
	} // namespace

	LameParameter<utils::ExpressionValue> lame_parameter_from_json(const json &j, const Units &units, const std::string &root_path)
	{
		bool has_E = j.contains("E");
		bool has_young = j.contains("young");
		bool has_nu = j.contains("nu");
		bool has_lambda = j.contains("lambda");
		bool has_mu = j.contains("mu");

		bool has_young_poisson = has_E || has_young || has_nu;
		bool has_lambda_mu = has_lambda || has_mu;

		const std::string stress = units.stress();
		LameParameter<utils::ExpressionValue> out;
		if (has_young_poisson)
		{
			out.type = LameParamType::YoungPoisson;
			out.coeff1 = parse_expr(j.at(has_young ? "young" : "E"), stress, root_path);
			out.coeff2 = parse_expr(j.at("nu"), "", root_path);
			return out;
		}

		if (has_lambda_mu)
		{
			out.type = LameParamType::LambdaMu;
			out.coeff1 = parse_expr(j.at("lambda"), stress, root_path);
			out.coeff2 = parse_expr(j.at("mu"), stress, root_path);
			return out;
		}

		throw std::runtime_error("LameParameter: expected young/nu, E/nu, or lambda/mu");
	}

} // namespace polyfem::material

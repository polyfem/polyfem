#include <polyfem/materials/Density.hpp>

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

	Density<utils::ExpressionValue> density_from_json(const json &j, const Units &units, const std::string &root_path)
	{
		Density<utils::ExpressionValue> out;
		if (j.contains("rho"))
		{
			out.rho = parse_expr(j.at("rho"), units.density(), root_path);
		}
		else
		{
			out.rho = parse_expr(j.at("density"), units.density(), root_path);
		}
		return out;
	}

} // namespace polyfem::material

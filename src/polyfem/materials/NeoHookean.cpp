#include <polyfem/materials/NeoHookean.hpp>

namespace polyfem::material
{
	NeoHookean<utils::ExpressionValue> neo_hookean_from_json(const json &j, const Units &units, const std::string &root_path)
	{
		NeoHookean<utils::ExpressionValue> out;
		out.lame = lame_parameter_from_json(j, units, root_path);
		return out;
	}

} // namespace polyfem::material

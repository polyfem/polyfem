#pragma once

#include <polyfem/Units.hpp>
#include <polyfem/materials/MaterialExprRegistry.hpp>
#include <polyfem/utils/ExpressionValue.hpp>

#include <nlohmann/json.hpp>
#include <string>

namespace polyfem::mesh
{
	class Mesh;
}

namespace polyfem::material
{
	/// @brief Build a material expression registry from the JSON material spec.
	/// @param materials Json material spec.
	/// @param units Unit for material expression.
	/// @param root_path Json root path for file path resolutiuon.
	MaterialExprRegistry build_material_expr_registry_from_json(
		const json &materials,
		const mesh::Mesh &mesh,
		const Units &units,
		const std::string &root_path);

}; // namespace polyfem::material

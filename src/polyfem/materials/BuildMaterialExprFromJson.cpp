#include <polyfem/materials/MaterialExprRegistry.hpp>

#include <polyfem/materials/BuildMaterialExprFromJson.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <nlohmann/json.hpp>

#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace polyfem::material
{
	namespace
	{
		// Dispatch from_json factory for a single material type.
		void dispatch_from_json(
			Span<const int> elements,
			const json &material,
			const Units &units,
			const std::string &root_path,
			MaterialExprRegistry &registry)
		{
			const std::string type = material.at("type").get<std::string>();

			// Sum material is handled one level higher. We only allow one level of material composition.
			if (type == "MaterialSum")
			{
				log_and_throw_error("Nested MaterialSum is not supported!");
			}
			else if (type == "NeoHookean")
				registry.set(elements, neo_hookean_from_json(material, units, root_path));
			else if (type == "Density" || type == "Dummy")
				return;
			else
				return;
		}

		void add_material_from_json(
			Span<const int> elements,
			const json &material,
			const Units &units,
			const std::string &root_path,
			MaterialExprRegistry &registry)
		{
			// MaterialSum is a json synatatic sugar to set multiple
			// materials to the same element group.
			if (material.at("type") == "MaterialSum")
			{
				for (const auto &model : material.value("models", json::array()))
				{
					dispatch_from_json(elements, model, units, root_path, registry);
				}
			}
			else
			{
				dispatch_from_json(elements, material, units, root_path, registry);
			}

			// Probably due to hisorically reason, density is an optional fields of material instead
			// of it's own dedicate material.
			if (material.contains("rho") || material.contains("density"))
				registry.set(elements, density_from_json(material, units, root_path));
		}
	} // namespace

	MaterialExprRegistry build_material_expr_registry_from_json(
		const json &materials,
		const mesh::Mesh &mesh,
		const Units &units,
		const std::string &root_path)
	{
		int n_elements = mesh.n_elements();
		assert(n_elements >= 0);
		MaterialExprRegistry registry{n_elements};
		// Legacy single-object mode applies one material to every element and ignores
		// its optional body id.
		if (!materials.is_array())
		{
			std::vector<int> elements(n_elements);
			std::iota(elements.begin(), elements.end(), 0);
			add_material_from_json(elements, materials, units, root_path, registry);
			return registry;
		}

		// In array mode, map each body to its last material block, matching the
		// legacy overwrite behavior for duplicate ids.
		std::unordered_map<int, int> body_id_to_material;
		for (int i = 0; i < materials.size(); ++i)
		{
			for (int id : utils::json_as_array<int>(materials[i].at("id")))
				body_id_to_material[id] = i;
		}

		std::vector<std::vector<int>> material_elements(materials.size());
		std::unordered_set<int> missing;
		for (int e = 0; e < n_elements; ++e)
		{
			int body_id = mesh.get_body_id(e);
			auto it = body_id_to_material.find(body_id);
			if (it == body_id_to_material.end())
			{
				missing.insert(body_id);
				continue;
			}
			material_elements[it->second].push_back(e);
		}

		for (int i = 0; i < materials.size(); ++i)
		{
			if (!material_elements[i].empty())
			{
				add_material_from_json(material_elements[i], materials[i], units, root_path, registry);
			}
		}

		for (int bid : missing)
		{
			logger().warn("Missing material parameters for body {}", bid);
		}

		return registry;
	}
} // namespace polyfem::material

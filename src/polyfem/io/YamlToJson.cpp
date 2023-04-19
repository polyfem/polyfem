// Original source code from https://github.com/mircodezorzi/tojson (MIT License)

#include "YamlToJson.hpp"

#include <yaml-cpp/yaml.h>

namespace polyfem::io
{
	namespace
	{
		inline nlohmann::json parse_scalar(const YAML::Node &node)
		{
			int i;
			double d;
			bool b;
			std::string s;

			if (YAML::convert<int>::decode(node, i))
				return i;
			if (YAML::convert<double>::decode(node, d))
				return d;
			if (YAML::convert<bool>::decode(node, b))
				return b;
			if (YAML::convert<std::string>::decode(node, s))
				return s;

			return nullptr;
		}

		/// @todo refactor and pass nlohmann::json down by reference instead of returning it
		inline nlohmann::json yaml_to_json(const YAML::Node &root)
		{
			nlohmann::json j{};

			switch (root.Type())
			{
			case YAML::NodeType::Null:
				break;
			case YAML::NodeType::Scalar:
				return parse_scalar(root);
			case YAML::NodeType::Sequence:
				for (auto &&node : root)
					j.emplace_back(yaml_to_json(node));
				break;
			case YAML::NodeType::Map:
				for (auto &&it : root)
					j[it.first.as<std::string>()] = yaml_to_json(it.second);
				break;
			default:
				break;
			}
			return j;
		}
	} // namespace

	json yaml_string_to_json(const std::string &yaml_str)
	{
		YAML::Node root = YAML::Load(yaml_str);
		return yaml_to_json(root);
	}

	json yaml_file_to_json(const std::string &yaml_filepath)
	{
		YAML::Node root = YAML::LoadFile(yaml_filepath);
		return yaml_to_json(root);
	}
} // namespace polyfem::io

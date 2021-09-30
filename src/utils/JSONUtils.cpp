#include <polyfem/JSONUtils.hpp>

#include <polyfem/StringUtils.hpp>
#include <polyfem/Logger.hpp>

#include <fstream>

namespace polyfem
{

	void apply_default_params(json &args)
	{
		assert(args.contains("default_params"));

		std::string default_params_path = resolve_path(args["default_params"], args["root_path"]);

		if (default_params_path.empty())
			return;

		std::ifstream file(default_params_path);
		if (!file.is_open())
		{
			logger().error("unable to open default params {} file", default_params_path);
			return;
		}

		json default_params;
		file >> default_params;
		file.close();

		default_params.merge_patch(args);
		args = default_params;
	}

} // namespace polyfem

#include "JSONUtils.hpp"

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <fstream>

#include <Eigen/Geometry>

namespace polyfem
{
	namespace utils
	{
		void apply_default_params(json &args)
		{
			assert(args.contains("common"));

			std::string default_params_path = resolve_path(args["common"], args["root_path"]);

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

		Eigen::Matrix3d to_rotation_matrix(const json &jr, std::string mode)
		{
			std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);

			Eigen::VectorXd r;
			if (jr.is_number())
			{
				r.setZero(3);
				assert(mode.size() == 1); // must be either "x", "y", or "z"
				int i = mode[0] - 'x';
				assert(i >= 0 && i < 3);
				r[i] = jr.get<double>();
			}
			else
			{
				assert(jr.is_array());
				r = jr;
			}

			if (mode == "axis_angle")
			{
				assert(r.size() == 4);
				double angle = deg2rad(r[0]); // NOTE: assumes input angle is in degrees
				Eigen::Vector3d axis = r.tail<3>().normalized();
				return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
			}

			if (mode == "quaternion")
			{
				assert(r.size() == 4);
				Eigen::Vector4d q = r.normalized();
				return Eigen::Quaterniond(q).toRotationMatrix();
			}

			// The following expect the input is given in degrees
			r = deg2rad(r);

			if (mode == "rotation_vector")
			{
				assert(r.size() == 3);
				double angle = r.norm();
				if (angle != 0)
				{
					return Eigen::AngleAxisd(angle, r / angle).toRotationMatrix();
				}
				else
				{
					return Eigen::Matrix3d::Identity();
				}
			}

			Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

			for (int i = 0; i < mode.size(); i++)
			{
				int j = mode[i] - 'x';
				assert(j >= 0 && j < 3);
				Eigen::Vector3d axis = Eigen::Vector3d::Zero();
				axis[j] = 1;
				R = Eigen::AngleAxisd(r[j], axis).toRotationMatrix() * R;
			}

			return R;
		}

		// check that incomming json doesn't have any unkown keys to avoid stupid bugs
		bool check_for_unknown_args(const json &args, const json &args_in, const std::string &path_prefix)
		{
			bool found_unknown_arg = false;
			json patch = json::diff(args, args_in);
			for (const json &op : patch)
			{
				if (op["op"].get<std::string>() == "add")
				{
					json::json_pointer new_path(op["path"].get<std::string>());
					if (!args_in[new_path.parent_pointer()].is_array()
						&& new_path.back().size() != 0
						&& new_path.back().front() != '#'
						&& new_path.back() != "authen_t1")
					{
						json::json_pointer parent = new_path.parent_pointer();
						logger().warn(
							"Unknown key in json (path={}{})",
							path_prefix, op["path"].get<std::string>());
						found_unknown_arg = true;
					}
				}
			}
			return found_unknown_arg;
		}

		bool is_param_valid(const json &params, const std::string &key)
		{
			return params.contains(key) && !params[key].is_null();
		}
	} // namespace utils
} // namespace polyfem

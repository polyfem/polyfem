#pragma once

#include <polyfem/Common.hpp>
#include <polysolve/JSONUtils.hpp>

#include <Eigen/Core>
#include <igl/PI.h>
#include <filesystem>

namespace polyfem
{
	namespace utils
	{
		void apply_common_params(json &args);

		// Templated degree to radians so a scalar or vector can be given
		template <typename T>
		inline T deg2rad(T deg)
		{
			return deg / 180 * igl::PI;
		}

		// Converts a JSON rotation expressed in the given rotation mode to a 3D rotation matrix.
		// NOTE: mode is a copy because the mode will be transformed to be case insensitive
		Eigen::Matrix3d to_rotation_matrix(const json &jr, std::string mode = "xyz");

		/// @brief Determine if a key exists and is non-null in a json object.
		/// @param params JSON of parameters
		/// @param key Key to check
		/// @return True if the key exists and is non-null, false otherwise.
		bool is_param_valid(const json &params, const std::string &key);

		/// @brief Return the value of a json object as an array.
		/// If the value is an array return it directly otherwise return a vector with the value.
		/// @param j JSON object
		/// @return Array of values
		template <typename T = json>
		std::vector<T> json_as_array(const json &j)
		{
			if (j.is_array())
			{
				return j.get<std::vector<T>>();
			}
			else
			{
				if constexpr (std::is_same_v<T, json>)
					return {j};
				else
					return {j.get<T>()};
			}
		}

		/// @brief Get a parameter from a json object or return a default value if the parameter invalid.
		/// Similar to json::value() but return default value if the parameter is null.
		template <typename T>
		T json_value(const json &params, const std::string &key, const T &default_value)
		{
			return is_param_valid(params, key) ? params[key].get<T>() : default_value;
		}
	} // namespace utils
} // namespace polyfem

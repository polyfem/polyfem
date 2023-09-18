#pragma once

#include <polyfem/Common.hpp>

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

namespace nlohmann
{
	template <typename T, int dim, int max_dim = dim>
	using Vector = Eigen::Matrix<T, dim, 1, Eigen::ColMajor, max_dim, 1>;
	template <typename T, int dim, int max_dim = dim>
	using RowVector = Eigen::Matrix<T, 1, dim, Eigen::RowMajor, 1, max_dim>;

	template <typename T, int dim, int max_dim>
	struct adl_serializer<Vector<T, dim, max_dim>>
	{
		static void to_json(json &j, const Vector<T, dim, max_dim> &v)
		{
			j = std::vector<T>(v.data(), v.data() + v.size());
		}

		static void from_json(const json &j, Vector<T, dim, max_dim> &v)
		{
			if (j.is_array())
			{
				auto jv = j.get<std::vector<T>>();
				v = Eigen::Map<Vector<T, dim, max_dim>>(jv.data(), long(jv.size()));
			}
			else if (j.is_number())
			{
				assert(dim == 1);
				v = Vector<T, 1>::Constant(j.get<T>());
			}
			else
			{
				assert(false);
			}
		}
	};

	template <typename T, int dim, int max_dim>
	struct adl_serializer<RowVector<T, dim, max_dim>>
	{
		static void to_json(json &j, const RowVector<T, dim, max_dim> &v)
		{
			j = std::vector<T>(v.data(), v.data() + v.size());
		}

		static void from_json(const json &j, RowVector<T, dim, max_dim> &v)
		{
			if (j.is_array())
			{
				auto jv = j.get<std::vector<T>>();
				v = Eigen::Map<Vector<T, dim, max_dim>>(jv.data(), long(jv.size()));
			}
			else if (j.is_number())
			{
				assert(dim == 1);
				v = RowVector<T, 1>::Constant(j.get<T>());
			}
			else
			{
				assert(false);
			}
		}
	};

	template <>
	struct adl_serializer<std::filesystem::path>
	{
		static void to_json(json &j, const std::filesystem::path &p)
		{
			j = p.string();
		}

		static void from_json(const json &j, std::filesystem::path &p)
		{
			p = j.get<std::string>();
		}
	};
} // namespace nlohmann

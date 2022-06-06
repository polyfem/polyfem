#pragma once

#include <polyfem/Common.hpp>

#include <igl/PI.h>

namespace polyfem
{
	namespace utils
	{
		void apply_default_params(json &args);

		// Templated degree to radians so a scalar or vector can be given
		template <typename T>
		inline T deg2rad(T deg)
		{
			return deg / 180 * igl::PI;
		}

		// Converts a JSON rotation expressed in the given rotation mode to a 3D rotation matrix.
		// NOTE: mode is a copy because the mode will be transformed to be case insensitive
		Eigen::Matrix3d to_rotation_matrix(const json &jr, std::string mode = "xyz");

		bool check_for_unknown_args(const json &args, const json &args_in, const std::string &path_prefix = "");

		bool is_param_valid(const json &params, const std::string &key);
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
			auto jv = j.get<std::vector<T>>();
			v = Eigen::Map<Vector<T, dim, max_dim>>(jv.data(), long(jv.size()));
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
			auto jv = j.get<std::vector<T>>();
			v = Eigen::Map<Vector<T, dim, max_dim>>(jv.data(), long(jv.size()));
		}
	};
} // namespace nlohmann

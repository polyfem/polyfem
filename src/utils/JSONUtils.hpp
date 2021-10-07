#pragma once

#include <polyfem/Common.hpp>

#include <igl/PI.h>

namespace polyfem
{

	void apply_default_params(json &args);

	// Templated degree to radians so a scalar or vector can be given
	template <typename T>
	inline T deg2rad(T deg)
	{
		return deg / 180 * igl::PI;
	}

	// Convert the given JSON array (or scalar) to a Eigen::Vector.
	template <typename Derived>
	void from_json(const json &j, Eigen::MatrixBase<Derived> &v)
	{
		auto jv = j.get<std::vector<typename Derived::Scalar>>();
		v = Eigen::Map<Derived>(jv.data(), long(jv.size()));
	}

	// Converts a JSON rotation expressed in the given rotation mode to a 3D rotation matrix.
	// NOTE: mode is a copy because the mode will be transformed to be case insensitive
	Eigen::Matrix3d to_rotation_matrix(const json &jr, std::string mode = "xyz");

} // namespace polyfem

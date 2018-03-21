#pragma once

#include <igl/AABB.h>
#include <igl/in_element.h>
#include <igl/barycentric_coordinates.h>

#include <Eigen/Dense>


namespace poly_fem
{
	class InterpolatedFunction2d
	{
	public:
		InterpolatedFunction2d() { }
		InterpolatedFunction2d(const Eigen::MatrixXd &fun, const Eigen::MatrixXd &pts, const Eigen::MatrixXi &tris);

		Eigen::MatrixXd interpolate(const Eigen::MatrixXd &pts) const;

	private:
		igl::AABB<Eigen::MatrixXd, 2> tree_;
		Eigen::MatrixXd fun_;
		Eigen::MatrixXd pts_;
		Eigen::MatrixXi tris_;
	};
}

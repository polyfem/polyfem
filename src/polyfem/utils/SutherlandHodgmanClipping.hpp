#pragma once

#include <vector>
#include <Eigen/Core>

namespace polyfem::utils
{
	/// Given a subject polygon defined by the vertices in clockwise order
	///
	/// subject_polygon = [(x_1,y_1),(x_2,y_2),...,(x_N,y_N)]
	///
	/// and a clipping polygon, which will be used to clip the subject polygon,
	/// defined by the vertices in clockwise order
	///
	/// clipping_polygon = [(x_1,y_1),(x_2,y_2),...,(x_K,y_K)]
	///
	/// and assuming that the subject polygon and clipping polygon overlap,
	/// the Sutherland-Hodgman algorithm finds the intersection of the two.
	///
	/// @warning Points must be in clockwise order or else this wont work.
	Eigen::MatrixXd sutherland_hodgman_clipping(
		const Eigen::MatrixXd &subject_polygon,
		const Eigen::MatrixXd &clipping_polygon);

}; // namespace polyfem::utils
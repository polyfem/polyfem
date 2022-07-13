#include "SutherlandHodgmanClipping.hpp"

namespace polyfem::utils
{
	namespace
	{
		bool is_inside(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2, const Eigen::Vector2d &q)
		{
			double R = (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0]);
			return R <= 0;
		}

		/// Given points p1 and p2 on line L1, compute the equation of L1 in the
		/// format of y = m1 * x + b1. Also, given points p3 and p4 on line L2,
		/// compute the equation of L2 in the format of y = m2 * x + b2.
		///
		/// To compute the point of intersection of the two lines, equate
		/// the two line equations together
		///
		/// m1 * x + b1 = m2 * x + b2
		///
		/// and solve for x. Once x is obtained, substitute it into one of the
		/// equations to obtain the value of y.
		///
		/// if one of the lines is vertical, then the x-coordinate of the point of
		/// intersection will be the x-coordinate of the vertical line. Note that
		/// there is no need to check if both lines are vertical (parallel), since
		/// this function is only called if we know that the lines intersect.
		Eigen::Vector2d compute_intersection(
			const Eigen::Vector2d &p1, const Eigen::Vector2d &p2,
			const Eigen::Vector2d &p3, const Eigen::Vector2d &p4)
		{

			double x, y;

			// if first line is vertical
			if (p2[0] - p1[0] == 0)
			{
				x = p1[0];

				// slope and intercept of second line
				const double m2 = (p4[1] - p3[1]) / (p4[0] - p3[0]);
				const double b2 = p3[1] - m2 * p3[0];

				// y-coordinate of intersection
				y = m2 * x + b2;
			}
			// if second line is vertical
			else if (p4[0] - p3[0] == 0)
			{
				x = p3[0];

				// slope and intercept of first line
				const double m1 = (p2[1] - p1[1]) / (p2[0] - p1[0]);
				const double b1 = p1[1] - m1 * p1[0];

				// y-coordinate of intersection
				y = m1 * x + b1;
			}
			// if neither line is vertical
			else
			{
				const double m1 = (p2[1] - p1[1]) / (p2[0] - p1[0]);
				const double b1 = p1[1] - m1 * p1[0];

				// slope and intercept of second line
				const double m2 = (p4[1] - p3[1]) / (p4[0] - p3[0]);
				const double b2 = p3[1] - m2 * p3[0];

				// x-coordinate of intersection
				x = (b2 - b1) / (m1 - m2);

				// y-coordinate of intersection
				y = m1 * x + b1;
			}

			return Eigen::Vector2d(x, y);
		}
	} // namespace

	// Given a subject polygon defined by the vertices in clockwise order
	//
	// subject_polygon = [(x_1,y_1),(x_2,y_2),...,(x_N,y_N)]
	//
	// and a clipping polygon, which will be used to clip the subject polygon,
	// defined by the vertices in clockwise order
	//
	// clipping_polygon = [(x_1,y_1),(x_2,y_2),...,(x_K,y_K)]
	//
	// and assuming that the subject polygon and clipping polygon overlap,
	// the Sutherland-Hodgman algorithm finds the intersection of the two.
	//
	// @warning Points must be in clockwise order or else this wont work.
	std::vector<Eigen::Vector2d> sutherland_hodgman_clipping(
		const Eigen::MatrixXd &subject_polygon,
		const Eigen::MatrixXd &clipping_polygon)
	{
		std::vector<Eigen::Vector2d> final_polygon(subject_polygon.rows());
		for (int i = 0; i < subject_polygon.rows(); ++i)
		{
			final_polygon[i] = subject_polygon.row(i);
		}

		for (int i = 0; i < clipping_polygon.rows(); ++i)
		{

			// stores the vertices of the next iteration of the clipping procedure
			std::vector<Eigen::Vector2d> next_polygon = final_polygon;

			// stores the vertices of the final clipped polygon
			final_polygon.clear();

			// these two vertices define a line segment (edge) in the clipping
			// polygon. It is assumed that indices wrap around, such that if
			// i = 1, then i - 1 = K.
			const Eigen::Vector2d &c_edge_start =
				clipping_polygon.row(i > 0 ? (i - 1) : clipping_polygon.rows() - 1);
			const Eigen::Vector2d &c_edge_end = clipping_polygon.row(i);

			for (int j = 0; j < next_polygon.size(); ++j)
			{
				// these two vertices define a line segment (edge) in the subject
				// polygon
				const Eigen::Vector2d &s_edge_start = next_polygon[j > 0 ? (j - 1) : next_polygon.size() - 1];
				const Eigen::Vector2d &s_edge_end = next_polygon[j];

				if (is_inside(c_edge_start, c_edge_end, s_edge_end))
				{
					if (!is_inside(c_edge_start, c_edge_end, s_edge_start))
					{
						final_polygon.push_back(compute_intersection(
							s_edge_start, s_edge_end, c_edge_start, c_edge_end));
					}
					final_polygon.push_back(s_edge_end);
				}
				else if (is_inside(c_edge_start, c_edge_end, s_edge_start))
				{
					final_polygon.push_back(compute_intersection(
						s_edge_start, s_edge_end, c_edge_start, c_edge_end));
				}
			}
		}

		Eigen::MatrixXd result(final_polygon.size(), 2);
		for (int i = 0; i < final_polygon.rows(); ++i)
		{
			result.row(i) = final_polygon[i];
		}
	}

}; // namespace polyfem::utils
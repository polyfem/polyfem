#include <polyfem/RefElementSampler.hpp>
#include <polyfem/MeshUtils.hpp>

#include <igl/triangle/triangulate.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/avg_edge_length.h>
#include <igl/edges.h>
#include <igl/write_triangle_mesh.h>


namespace poly_fem {

namespace {

///
/// Generate a canonical triangle subdivided into smaller triangles with the
/// target area. This also ensures that boundary is periodic (boundary edges are
/// evenly subdivided).
///
/// @param[in]  target_area  { Target triangle area }
/// @param[out] OV           { #V x 2 output vertices positions }
/// @param[out] OF           { #F x 3 output triangle indices }
///
void triangulate_periodic(double target_area, Eigen::MatrixXd &OV, Eigen::MatrixXi &OF, Eigen::MatrixXi &OE) {
	std::stringstream buf;
	buf.precision(100);
	buf.setf(std::ios::fixed, std::ios::floatfield);
	buf << "Qqa" << target_area;

	// Equilateral triangle
	Eigen::MatrixXd V(3, 2);
	V << 0, 0,
		1, 0,
		0.5, 0.5 * std::sqrt(3.0);

	Eigen::MatrixXi E(3, 2);
	E << 0, 1,
		1, 2,
		2, 0;

	// 1st pass gets an rough idea
	igl::triangle::triangulate(V, E, Eigen::MatrixXd(0,2), buf.str(), OV, OF);

	// Extract average edge-length, use it to sample the boundary evenly
	double avg_len = igl::avg_edge_length(OV, OF);
	int n = 1.0 / avg_len;

	// Build subdivided triangle boundary
	Eigen::MatrixXd V2(3 * n, 2);
	Eigen::MatrixXi E2(3 * n, 2);
	int rows = 3 * n;
	int cols = 2;
	E2 = Eigen::VectorXi::LinSpaced(rows, 0.0, rows - 1).replicate(1, cols);
	E2.col(1) = E2.col(1).unaryExpr([&](const int x) { return (x+1)%rows; });
	for (int d = 0; d < V2.cols(); ++d) {
		V2.topRows(n).col(d) = Eigen::VectorXd::LinSpaced(n + 1, V(0, d), V(1, d)).head(n);
		V2.middleRows(n, n).col(d) = Eigen::VectorXd::LinSpaced(n + 1, V(1, d), V(2, d)).head(n);
		V2.bottomRows(n).col(d) = Eigen::VectorXd::LinSpaced(n + 1, V(2, d), V(0, d)).head(n);
	}

	// 2nd pass disable Steiner points on the boundary
	buf << "Y";
	igl::triangle::triangulate(V2, E2, Eigen::MatrixXd(0,2), buf.str(), OV, OF);
	OE = E2;

	// Warp vertex positions to map them back to the canonical element
	Eigen::Matrix2d M, Minv;
	M << 1, 0,
		0.5, 0.5 * std::sqrt(3.0);
	Minv = M.inverse();
	OV = OV * Minv;
}

} // anonymous namespace

	RefElementSampler &RefElementSampler::sampler()
	{
		static RefElementSampler instance;

		return instance;
	}

	void RefElementSampler::init(const bool is_volume, const int n_elements, double target_rel_area)
	{
		is_volume_ = is_volume;

		area_param_ = target_rel_area * n_elements;
#ifndef NDEBUG
		area_param_ *= 10.0;
#endif

		build();
	}

	void RefElementSampler::build()
	{
		using namespace Eigen;

		std::stringstream buf;
		buf.precision(100);
		buf.setf(std::ios::fixed, std::ios::floatfield);

		if(is_volume_)
		{
			buf<<"Qpq1.414a"<<area_param_;
			{
				MatrixXd pts(8,3); pts <<
				0, 0, 0,
				0, 1, 0,
				1, 1, 0,
				1, 0, 0,

				//4
				0, 0, 1,
				0, 1, 1,
				1, 1, 1,
				1, 0, 1;

				Eigen::MatrixXi faces(12,3); faces <<
				1, 2, 0,
				0, 2, 3,

				5, 4, 6,
				4, 7, 6,

				1, 0, 4,
				1, 4, 5,

				2, 1, 5,
				2, 5, 6,

				3, 2, 6,
				3, 6, 7,

				0, 3, 7,
				0, 7, 4;

				igl::copyleft::tetgen::tetrahedralize(pts, faces, buf.str(), cube_points_, cube_tets_, cube_faces_);

				// Extract sampled edges matching the base element edges
				Eigen::MatrixXi edges(12, 2); edges <<
				0, 1,
				1, 2,
				2, 3,
				3, 0,

				4, 5,
				5, 6,
				6, 7,
				7, 4,

				0, 4,
				1, 5,
				2, 6,
				3, 7;
				igl::edges(cube_faces_, cube_edges_);
				extract_parent_edges(cube_points_, cube_edges_, pts, edges, cube_edges_);
			}
			{
				MatrixXd pts(4,3); pts <<
				0, 0, 0,
				1, 0, 0,
				0, 1, 0,
				0, 0, 1;

				Eigen::MatrixXi faces(4,3); faces <<
				0, 1, 2,

				3, 1, 0,
				2, 1, 3,
				0, 2, 3;

				MatrixXi tets;
				igl::copyleft::tetgen::tetrahedralize(pts, faces, buf.str(), simplex_points_, simplex_tets_, simplex_faces_);

				// Extract sampled edges matching the base element edges
				Eigen::MatrixXi edges;
				igl::edges(faces, edges);
				igl::edges(simplex_faces_, simplex_edges_);
				extract_parent_edges(simplex_points_, simplex_edges_, pts, edges, simplex_edges_);
			}
		}
		else
		{
			buf<<"Qqa"<<area_param_;
			{
				MatrixXd pts(4,2); pts <<
				0,0,
				0,1,
				1,1,
				1,0;

				MatrixXi E(4,2); E <<
				0,1,
				1,2,
				2,3,
				3,0;

				MatrixXd H(0,2);
				igl::triangle::triangulate(pts, E, H, buf.str(), cube_points_, cube_faces_);

				// Extract sampled edges matching the base element edges
				igl::edges(cube_faces_, cube_edges_);
				extract_parent_edges(cube_points_, cube_edges_, pts, E, cube_edges_);
			}
			{
				triangulate_periodic(area_param_, simplex_points_, simplex_faces_, simplex_edges_);
			}
		}

	}

}

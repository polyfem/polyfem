#include <polyfem/RefElementSampler.hpp>
#include <polyfem/MeshUtils.hpp>

#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/avg_edge_length.h>
#include <igl/edges.h>
#include <igl/write_triangle_mesh.h>

namespace polyfem
{

namespace
{
///
/// Generate a canonical triangle subdivided fomr a regular grid
///
/// @param[in]  n  			 { n grid quads }
/// @param[in]  tri			 { is a tri or a quad }
/// @param[out] OV           { #V x 2 output vertices positions }
/// @param[out] OF           { #F x 3 output triangle indices }
///
void regular_2d_grid(const int n, bool tri, Eigen::MatrixXd &V, Eigen::MatrixXi &F)
{

	V.resize(n * n, 2);
	F.resize((n - 1) * (n - 1) * 2, 3);
	double delta = 1. / (n - 1.);
	std::vector<int> map(n * n, -1);

	int index = 0;
	for(int i=0; i < n; ++i)
	{
    	for(int j = 0; j < n; ++j)
		{
        	if(tri && i+j >= n)
            	continue;
        map[i*n+j]=index;
		V.row(index) << i * delta, j * delta;
		++index;
		}
	}

	V.conservativeResize(index, 2);

	std::array<int, 3> tmp;

	index = 0;
	for(int i=0; i < n-1; ++i)
	{
    	for(int j = 0; j < n-1; ++j)
		{
        	tmp = {{ map[i + j*n], map[i+1 + j*n], map[i + (j+1)*n] }};
			if(tmp[0] >= 0 && tmp[1] >= 0 && tmp[2] >= 0)
			{
				F.row(index) << tmp[0], tmp[1], tmp[2];
				++index;
			}

			tmp = {{ map[ i + 1 + j * n], map[i + 1 + (j + 1) * n], map[i + (j + 1) * n ] }};
			if (tmp[0] >= 0 && tmp[1] >= 0 && tmp[2] >= 0) {
				F.row(index) << tmp[0], tmp[1], tmp[2];
				++index;
			}
		}
	}

	F.conservativeResize(index, 3);
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

	if (is_volume_)
	{
		buf << "Qpq1.414a" << area_param_;
		{
			MatrixXd pts(8, 3);
			pts << 0, 0, 0,
				0, 1, 0,
				1, 1, 0,
				1, 0, 0,

				//4
				0, 0, 1,
				0, 1, 1,
				1, 1, 1,
				1, 0, 1;

			Eigen::MatrixXi faces(12, 3);
			faces << 1, 2, 0,
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
			Eigen::MatrixXi edges(12, 2);
			edges << 0, 1,
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

			// Same local order as in FEMBasis3d
			cube_corners_.resize(8, 3);
			cube_corners_ << 0, 0, 0,
				1, 0, 0,
				1, 1, 0,
				0, 1, 0,
				0, 0, 1,
				1, 0, 1,
				1, 1, 1,
				0, 1, 1;
		}
		{
			MatrixXd pts(4, 3);
			pts << 0, 0, 0,
				1, 0, 0,
				0, 1, 0,
				0, 0, 1;

			Eigen::MatrixXi faces(4, 3);
			faces << 0, 1, 2,

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

			// Same local order as in FEMBasis3d
			simplex_corners_.resize(4, 3);
			simplex_corners_ << 0, 0, 0,
				1, 0, 0,
				0, 1, 0,
				0, 0, 1;
		}
	}
	else
	{
		{
			MatrixXd pts(4, 2);
			pts << 0, 0,
				0, 1,
				1, 1,
				1, 0;

			MatrixXi E(4, 2);
			E << 0, 1,
				1, 2,
				2, 3,
				3, 0;

			regular_2d_grid(1. / sqrt(area_param_) + 1, false, cube_points_, cube_faces_);

			// Extract sampled edges matching the base element edges
			igl::edges(cube_faces_, cube_edges_);
			extract_parent_edges(cube_points_, cube_edges_, pts, E, cube_edges_);

			// Same local order as in FEMBasis2d
			cube_corners_.resize(4, 2);
			cube_corners_ << 0, 0,
				1, 0,
				1, 1,
				0, 1;
		}
		{
			MatrixXd pts(3, 2);
			pts << 0, 0,
				0, 1,
				1, 0;

			MatrixXi E(3, 2);
			E << 0, 1,
				1, 2,
				2, 0;

			regular_2d_grid(1. / sqrt(area_param_) + 1, true, simplex_points_, simplex_faces_);

			// Extract sampled edges matching the base element edges
			igl::edges(simplex_faces_, simplex_edges_);
			extract_parent_edges(simplex_points_, simplex_edges_, pts, E, simplex_edges_);

			// Same local order as in FEMBasis2d
			simplex_corners_.resize(3, 2);
			simplex_corners_ << 0, 0,
				1, 0,
				0, 1;
		}
	}
}

void RefElementSampler::sample_polygon(const Eigen::MatrixXd &poly, Eigen::MatrixXd &pts, Eigen::MatrixXi &faces) const
{
	// MatrixXi E(poly.rows(), 2);
	// for (int e = 0; e < int(poly.rows()); ++e)
	// {
	// 	E(e, 0) = e;
	// 	E(e, 1) = (e + 1) % poly.rows();
	// }
	// std::stringstream buf;
	// buf.precision(100);
	// buf.setf(std::ios::fixed, std::ios::floatfield);
	// buf << "Qqa" << area_param_ / 1000.;

	// igl::triangle::triangulate(poly, E, MatrixXd(0, 2), buf.str(), pts, faces);
	pts.resize(poly.rows()+1, poly.cols());
	pts.block(0, 0, poly.rows(), poly.cols()) = poly;
	pts.row(poly.rows()) = poly.colwise().mean();

	faces.resize(poly.rows(), 3);
	for (int e = 0; e < int(poly.rows()); ++e)
	{
		faces.row(e) << e , (e + 1) % poly.rows(), poly.rows();
	}
}

void RefElementSampler::sample_polyhedron(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &f, Eigen::MatrixXd &pts, Eigen::MatrixXi &faces) const
{
	//TODO
	pts = vertices;
	faces = f;
}

} // namespace polyfem

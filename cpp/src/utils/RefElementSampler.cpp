#include "RefElementSampler.hpp"
#include "MeshUtils.hpp"

#include <igl/triangle/triangulate.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/edges.h>


namespace poly_fem
{

	RefElementSampler &RefElementSampler::sampler()
	{
		static RefElementSampler instance;

		return instance;
	}

	void RefElementSampler::init(const bool is_volume, const int n_elements)
	{
		is_volume_ = is_volume;

#ifdef NDEBUG
		area_param_ = 0.00001*n_elements;
#else
		area_param_ = 0.0001*n_elements;
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
			}
			{
				MatrixXd pts(3,2); pts <<
				0,0,
				1,0,
				0,1;

				MatrixXi E(3,2); E <<
				0,1,
				1,2,
				2,0;

				igl::triangle::triangulate(pts, E, MatrixXd(0,2), buf.str(), simplex_points_, simplex_faces_);

				// Extract sampled edges matching the base element edges
				igl::edges(simplex_faces_, simplex_edges_);
				Eigen::MatrixXd tmp = simplex_points_;
				tmp.conservativeResize(tmp.rows(), 3);
				tmp.col(2).setZero();

				Eigen::MatrixXd tmp1 = pts;
				tmp1.conservativeResize(tmp1.rows(), 3);
				tmp1.col(2).setZero();
				extract_parent_edges(tmp, simplex_edges_, tmp1, E, simplex_edges_);
			}
		}

	}

}

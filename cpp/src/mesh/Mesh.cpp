#include "Mesh.hpp"

#include <igl/triangle/triangulate.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

#include <geogram/basic/file_system.h>
#include <geogram/mesh/mesh_io.h>

#include <cassert>

namespace poly_fem
{
	bool Mesh::load(const std::string &path)
	{
		mesh_.clear(false,false);

		if(!mesh_load(path, mesh_))
			return false;

		return true;
	}

	bool Mesh::save(const std::string &path) const
	{
		if(!mesh_save(mesh_, path))
			return false;

		return true;
	}

	void Mesh::point(const int global_index, Eigen::MatrixXd &pt) const
	{
		pt.resize(1, is_volume() ? 3 : 2);
		const double *pt_ptr = mesh_.vertices.point_ptr(global_index);
		pt(0) = pt_ptr[0];
		pt(1) = pt_ptr[1];

		if(is_volume())
			pt(2) = pt_ptr[2];
	}

	void Mesh::triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts) const
	{
		if(is_volume())
		{
			//TODO
			assert(false);
		}
		else
		{
			std::vector<Eigen::MatrixXi> local_tris(mesh_.facets.nb());
			std::vector<Eigen::MatrixXd> local_pts(mesh_.facets.nb());

			int total_tris = 0;
			int total_pts  = 0;

			for(GEO::index_t f = 0; f < mesh_.facets.nb(); ++f)
			{
				const int n_vertices = mesh_.facets.nb_vertices(f);

				Eigen::MatrixXd face_pts(n_vertices, 2);
				Eigen::MatrixXi edges(n_vertices,2);

				for(int i = 0; i < n_vertices; ++i)
				{
					const int vertex = mesh_.facets.vertex(f,i);
					const double *pt = mesh_.vertices.point_ptr(vertex);
					face_pts(i, 0) = pt[0];
					face_pts(i, 1) = pt[1];

					edges(i, 0) = i;
					edges(i, 1) = (i+1) % n_vertices;
				}

				igl::triangle::triangulate(face_pts, edges, Eigen::MatrixXd(0,2), "QqYS0", local_pts[f], local_tris[f]);

				total_tris += local_tris[f].rows();
				total_pts  += local_pts[f].rows();

				assert(local_pts[f].rows() == face_pts.rows());
			}


			tris.resize(total_tris, 3);
			pts.resize(total_pts, 2);

			int tri_index = 0;
			int pts_index = 0;
			for(std::size_t i = 0; i < local_tris.size(); ++i){
				tris.block(tri_index, 0, local_tris[i].rows(), local_tris[i].cols()) = local_tris[i].array() + pts_index;
				tri_index += local_tris[i].rows();

				pts.block(pts_index, 0, local_pts[i].rows(), local_pts[i].cols()) = local_pts[i];
				pts_index += local_pts[i].rows();
			}
		}
	}
}
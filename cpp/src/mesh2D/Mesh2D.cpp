#include <polyfem/Mesh2D.hpp>
#include <polyfem/Navigation.hpp>

#include <polyfem/MeshUtils.hpp>
#include <polyfem/Refinement.hpp>

#include <igl/triangle/triangulate.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

#include <geogram/basic/file_system.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram/mesh/mesh_repair.h>

#include <cassert>
#include <array>

namespace poly_fem
{
	void Mesh2D::refine(const int n_refinement, const double t, std::vector<int> &parent_nodes)
	{
		// return;
		if (n_refinement <= 0) { return; }

		bool all_simplicial = true;
		for (int e = 0; e < n_elements(); ++e) {
			all_simplicial &= is_simplex(e);
		}

		for (int i = 0; i < n_refinement; ++i)
		{
			GEO::Mesh mesh;
			mesh.copy(mesh_);
			mesh_.clear(false,false);

			//TODO add tags to the refinement
			if (all_simplicial) {
				refine_triangle_mesh(mesh, mesh_);
			} else if (t<=0) {
				refine_polygonal_mesh(mesh, mesh_, Polygons::catmul_clark_split_func());
			} else {
				refine_polygonal_mesh(mesh, mesh_, Polygons::polar_split_func(t));
			}

			Navigation::prepare_mesh(mesh_);
		}

		compute_elements_tag();

		// save("test.obj");
	}

	bool Mesh2D::load(const std::string &path)
	{
		mesh_.clear(false,false);

		if(!mesh_load(path, mesh_))
			return false;

		orient_normals_2d(mesh_);
		Navigation::prepare_mesh(mesh_);
		compute_elements_tag();
		return true;
	}

	bool Mesh2D::load(const GEO::Mesh &mesh)
	{
		mesh_.clear(false,false);
		mesh_.copy(mesh);

		orient_normals_2d(mesh_);
		Navigation::prepare_mesh(mesh_);
		compute_elements_tag();
		return true;
	}

	bool Mesh2D::save(const std::string &path) const
	{
		if(!mesh_save(mesh_, path))
			return false;

		return true;
	}

	void Mesh2D::bounding_box(RowVectorNd &min, RowVectorNd &max) const
	{
		GEO::vec3 min_corner, max_corner;
		GEO::get_bbox(mesh_, &min_corner[0], &max_corner[0]);
		min.resize(2); max.resize(2);

		min(0) = min_corner.x;
		min(1) = min_corner.y;

		max(0) = max_corner.x;
		max(1) = max_corner.y;
	}

	void Mesh2D::normalize() {

		GEO::vec3 min_corner, max_corner;
		GEO::get_bbox(mesh_, &min_corner[0], &max_corner[0]);
		GEO::vec3 extent = max_corner - min_corner;
		double scaling = std::max(extent[0], std::max(extent[1], extent[2]));
		// const GEO::vec3 origin = 0.5 * (min_corner + max_corner);
		const GEO::vec3 origin = min_corner;
		for (GEO::index_t v = 0; v < mesh_.vertices.nb(); ++v) {
			mesh_.vertices.point(v) = (mesh_.vertices.point(v) - origin) / scaling;
		}

		std::cout << "-- bbox before normalization:" << std::endl;
		std::cout << "   min   : " << min_corner << std::endl;
		std::cout << "   max   : " << max_corner << std::endl;
		std::cout << "   extent: " << max_corner - min_corner << std::endl;
		std::cout << "-- bbox after normalization:" << std::endl;
		GEO::get_bbox(mesh_, &min_corner[0], &max_corner[0]);
		std::cout << "   min   : " << min_corner << std::endl;
		std::cout << "   max   : " << max_corner << std::endl;
		std::cout << "   extent: " << max_corner - min_corner << std::endl;

		Eigen::MatrixXd p0, p1, p;
		get_edges(p0, p1);
		p = p0 - p1;
		std::cout << "-- edge length after normalization:" << std::endl;
		std::cout << "   min: " << p.rowwise().norm().minCoeff() << std::endl;
		std::cout << "   max: " << p.rowwise().norm().maxCoeff() << std::endl;
		std::cout << "   avg: " << p.rowwise().norm().mean() << std::endl;
	}


	double Mesh2D::edge_length(const int gid) const
	{
		const int v0 = mesh_.edges.vertex(gid, 0);
		const int v1 = mesh_.edges.vertex(gid, 1);

		return (point(v0) - point(v1)).norm();
	}

	void Mesh2D::get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const
	{
		p0.resize(mesh_.edges.nb(), 2);
		p1.resize(p0.rows(), p0.cols());

		for(GEO::index_t e = 0; e < mesh_.edges.nb(); ++e)
		{
			const int v0 = mesh_.edges.vertex(e, 0);
			const int v1 = mesh_.edges.vertex(e, 1);

			p0.row(e) = point(v0);
			p1.row(e) = point(v1);
		}
	}

	void Mesh2D::get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1, const std::vector<bool> &valid_elements) const
	{
		int count = 0;
		for(size_t i = 0; i < valid_elements.size(); ++i)
		{
			if(valid_elements[i]){
				count += n_face_vertices(i);
			}
		}

		p0.resize(count, 2);
		p1.resize(count, 2);

		count = 0;

		for(size_t i = 0; i < valid_elements.size(); ++i)
		{
			if(!valid_elements[i])
				continue;

			auto index = get_index_from_face(i);
			for(int j = 0; j < n_face_vertices(i); ++j)
			{
				p0.row(count) = point(index.vertex);
				p1.row(count) = point(switch_vertex(index).vertex);

				index = next_around_face(index);
				++count;
			}
		}
	}

	void Mesh2D::set_point(const int global_index, const RowVectorNd &p)
	{
		mesh_.vertices.point(global_index).x = p(0);
		mesh_.vertices.point(global_index).y = p(1);
	}

	RowVectorNd Mesh2D::point(const int global_index) const {
		const double *ptr = mesh_.vertices.point_ptr(global_index);
		RowVectorNd p(2);
		p(0) = ptr[0];
		p(1) = ptr[1];
		return p;
	}

	void Mesh2D::compute_boundary_ids(const std::function<int(const RowVectorNd&)> &marker)
	{
		boundary_ids_.resize(n_edges());
		std::fill(boundary_ids_.begin(), boundary_ids_.end(), -1);

		//implement me properly
		for(int e = 0; e < n_edges(); ++e)
		{
			if(!is_boundary_edge(e))
				continue;

			const auto p = edge_barycenter(e);

			boundary_ids_[e] = marker(p);
		}
	}

	void Mesh2D::compute_boundary_ids()
	{
		boundary_ids_.resize(n_edges());
		std::fill(boundary_ids_.begin(), boundary_ids_.end(), -1);

		GEO::vec3 min_corner, max_corner;
		GEO::get_bbox(mesh_, &min_corner[0], &max_corner[0]);

		//implement me properly
		for(int e = 0; e < n_edges(); ++e)
		{
			if(!is_boundary_edge(e))
				continue;

			const auto p = edge_barycenter(e);

			if(fabs(p(0)-min_corner[0])<1e-7)
				boundary_ids_[e]=1;
			if(fabs(p(1)-min_corner[1])<1e-7)
				boundary_ids_[e]=2;
			if(fabs(p(0)-max_corner[0])<1e-7)
				boundary_ids_[e]=3;
			if(fabs(p(1)-max_corner[1])<1e-7)
				boundary_ids_[e]=4;
		}
	}


	void Mesh2D::triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const
	{
		ranges.clear();

		std::vector<Eigen::MatrixXi> local_tris(mesh_.facets.nb());
		std::vector<Eigen::MatrixXd> local_pts(mesh_.facets.nb());

		int total_tris = 0;
		int total_pts  = 0;


		ranges.push_back(0);

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

			ranges.push_back(total_tris);

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

	void Mesh2D::compute_elements_tag()
	{
		elements_tag_.clear();
		poly_fem::compute_element_tags(mesh_, elements_tag_);
	}

	void Mesh2D::update_elements_tag()
	{
		poly_fem::compute_element_tags(mesh_, elements_tag_);
	}

	RowVectorNd Mesh2D::edge_barycenter(const int index) const
	{
		const int v0 = mesh_.edges.vertex(index, 0);
		const int v1 = mesh_.edges.vertex(index, 1);

		return 0.5*(point(v0) + point(v1));
	}

	RowVectorNd Mesh2D::face_barycenter(const int face_index) const
	{
		RowVectorNd bary(2); bary.setZero();

		const int n_vertices = n_face_vertices(face_index);
		Navigation::Index index = get_index_from_face(face_index);

		for(int lv = 0; lv < n_vertices; ++lv)
		{
			bary += point(index.vertex);
			index = next_around_face(index);
		}
		return bary / n_vertices;
	}

}

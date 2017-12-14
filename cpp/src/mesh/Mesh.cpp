#include "Mesh.hpp"
#include "Navigation.hpp"

#include <igl/triangle/triangulate.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>

#include <geogram/basic/file_system.h>
#include <geogram/mesh/mesh_io.h>

#include <cassert>
#include <array>

namespace poly_fem
{
	bool Mesh::load(const std::string &path)
	{
		mesh_.clear(false,false);

		if(!mesh_load(path, mesh_))
			return false;

		Navigation::prepare_mesh(mesh_);
		create_boundary_nodes();
		return true;
	}

	bool Mesh::save(const std::string &path) const
	{
		if(!mesh_save(mesh_, path))
			return false;

		return true;
	}

	void Mesh::get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1)
	{
		p0.resize(mesh_.edges.nb(), is_volume() ? 3 : 2);
		p1.resize(p0.rows(), p0.cols());

		Eigen::MatrixXd p0t, p1t;
		for(GEO::index_t e = 0; e < mesh_.edges.nb(); ++e)
		{
			const int v0 = mesh_.edges.vertex(e, 0);
			const int v1 = mesh_.edges.vertex(e, 1);

			point(v0, p0t); point(v1, p1t);

			p0.row(e) = p0t;
			p1.row(e) = p1t;
		}
	}

	double Mesh::compute_mesh_size() const
	{
		Eigen::MatrixXd p0, p1, p;
		double sum = 0;
		for(GEO::index_t e = 0; e < mesh_.edges.nb(); ++e)
		{
			const int v0 = mesh_.edges.vertex(e, 0);
			const int v1 = mesh_.edges.vertex(e, 1);
			point(v0, p0); point(v1, p1);

			p = p0-p1;
			sum += p.norm();
		}

		return sum/double(mesh_.edges.nb());
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

	void Mesh::set_boundary_tags(std::vector<int> &tags) const
	{
		if(is_volume())
		{
			assert(false);
		}
		else
		{
			tags.resize(mesh_.edges.nb());
			std::fill(tags.begin(), tags.end(), -1);

			Eigen::MatrixXd p0, p1, p;

			const GEO::Attribute<int> boundary(mesh_.edges.attributes(), "boundary_edge");
			for(GEO::index_t e = 0; e < mesh_.edges.nb(); ++e)
			{
				if(boundary[e] != 1)
					continue;

				const int v0 = mesh_.edges.vertex(e, 0);
				const int v1 = mesh_.edges.vertex(e, 1);
				point(v0, p0); point(v1, p1);

				p = (p0 + p1)/2;

				if(fabs(p(0))<1e-8)
					tags[e]=1;
				if(fabs(p(1))<1e-8)
					tags[e]=2;
				if(fabs(p(0)-1)<1e-8)
					tags[e]=3;
				if(fabs(p(1)-1)<1e-8)
					tags[e]=4;
			}
		}
	}

	Navigation::Index Mesh::get_index_from_face(int f, int lv) const
	{
		return Navigation::get_index_from_face(mesh_, f, lv);
	}


	Navigation::Index Mesh::switch_vertex(Navigation::Index idx) const
	{
		return Navigation::switch_vertex(mesh_, idx);
	}

	Navigation::Index Mesh::switch_edge(Navigation::Index idx) const
	{
		return Navigation::switch_edge(mesh_, idx);
	}

	Navigation::Index Mesh::switch_face(Navigation::Index idx) const
	{
		return Navigation::switch_face(mesh_, idx);
	}

	Eigen::MatrixXd Mesh::node_from_face(const int face_id) const
	{
		Eigen::MatrixXd res=Eigen::MatrixXd::Zero(1, is_volume()? 3:2);
		Eigen::MatrixXd pt;

		for(int i = 0; i < n_element_vertices(face_id); ++i)
		{
			point(vertex_global_index(face_id, i), pt);
			res += pt;
		}

		return res / n_element_vertices(face_id);
	}

	Eigen::MatrixXd Mesh::node_from_edge_index(const Navigation::Index &index) const
	{
		int id = switch_face(index).face;
		if(id >= 0)
		{
			if(mesh_.facets.nb_vertices(id) == 4)
				return node_from_face(id);
		}

		id = edge_node_id(index.edge);
		assert(id >= 0);

		GEO::Attribute<std::array<double, 3> > edges_node(mesh_.edges.attributes(), "edges_node");
		Eigen::MatrixXd res(1, is_volume()? 3:2);

		for(long i = 0; i < res.size(); ++i)
			res(i) = edges_node[index.edge][i];

		return res;
	}

	Eigen::MatrixXd Mesh::node_from_vertex(const int vertex_id) const
	{
		GEO::Attribute<int> vertices_node_id(mesh_.vertices.attributes(), "vertices_node_id");
		assert(vertices_node_id[vertex_id] >= 0);

		GEO::Attribute<std::array<double, 3>> vertices_node(mesh_.vertices.attributes(), "vertices_node");
		Eigen::MatrixXd res(1, is_volume()? 3:2);
		for(long i = 0; i < res.size(); ++i)
			res(i) = vertices_node[vertex_id][i];

		return res;
	}

	int Mesh::edge_node_id(const int edge_id) const
	{
		GEO::Attribute<int> edges_node_id(mesh_.edges.attributes(), "edges_node_id");
		return edges_node_id[edge_id];
	}

	int Mesh::vertex_node_id(const int vertex_id) const
	{
		GEO::Attribute<int> vertices_node_id(mesh_.vertices.attributes(), "vertices_node_id");
		return vertices_node_id[vertex_id];
	}

	bool Mesh::node_id_from_edge_index(const Navigation::Index &index, int &id) const
	{
		id = switch_face(index).face;
		bool is_real_boundary = true;
		if(id >= 0)
		{
			is_real_boundary = false;
			if(mesh_.facets.nb_vertices(id) == 4)
				return is_real_boundary;

		}

		id = edge_node_id(index.edge);
		assert(id >= 0);

		return is_real_boundary;
	}

	void Mesh::element_bounday_polygon(const int index, Eigen::MatrixXd &poly) const
	{
		if(is_volume())
		{
			assert(false);
		}
		else
		{
			poly.resize(mesh_.facets.nb_vertices(index), 2);
			Eigen::MatrixXd p;

			for(GEO::index_t i = 0; i < mesh_.facets.nb_vertices(index); ++i)
			{
				const GEO::index_t vid = mesh_.facets.vertex(index, i);
				point(vid, p);
				poly.row(i) = p;
			}
		}
	}

	void Mesh::create_boundary_nodes()
	{
		GEO::Attribute<int> boundary(mesh_.edges.attributes(), "boundary_edge");

		for(GEO::index_t f = 0; f < mesh_.facets.nb(); ++f)
		{
			const int n_vertices = mesh_.facets.nb_vertices(f);
			if(n_vertices <= 4) continue;

			Navigation::Index index = get_index_from_face(f);

			for(int j = 0; j < n_vertices; ++j)
			{
				if(boundary[index.edge] == 0)
					boundary[index.edge] = 2;

				index = next_around_face(index);
			}
		}

		GEO::Attribute<int> edges_node_id(mesh_.edges.attributes(), "edges_node_id");
		GEO::Attribute<std::array<double, 3> > edges_node(mesh_.edges.attributes(), "edges_node");

		std::vector<int> vertex_counter(n_pts(), 0);

		int counter = n_elements();

		Eigen::MatrixXd p0, p1;

		for (int e = 0; e < (int) mesh_.edges.nb(); ++e)
		{
			if(boundary[e] == 0)
			{
				edges_node_id[e] = -1;
				continue;
			}

			edges_node_id[e] = counter++;

			const int v0 = mesh_.edges.vertex(e, 0);
			const int v1 = mesh_.edges.vertex(e, 1);
			point(v0, p0); point(v1, p1);
			auto &val = (p0 + p1)/2;
			for(long d = 0; d < val.size(); ++d)
				edges_node[e][d] = val(d);
		}

		GEO::Attribute<int> vertices_node_id(mesh_.vertices.attributes(), "vertices_node_id");
		vertices_node_id.fill(-1);

		GEO::Attribute<std::array<double, 3>> vertices_node(mesh_.vertices.attributes(), "vertices_node");

		for (int e = 0; e < n_elements(); ++e)
		{
			Navigation::Index index = get_index_from_face(e);

			bool was_boundary = boundary[get_index_from_face(e, n_element_vertices(e)-1).edge] != 0;
			for(int i = 0; i < n_element_vertices(e); ++i)
			{
				if(was_boundary)
				{
					if(boundary[index.edge] != 0)
					{
						const int v_id = index.vertex;
						vertices_node_id[v_id] = counter++;
						point(v_id, p0);

						for(long d = 0; d < p0.size(); ++d)
							vertices_node[v_id][d] = p0(d);
					}
				}

				was_boundary = boundary[index.edge] != 0;
				index = next_around_face(index);
			}
		}
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

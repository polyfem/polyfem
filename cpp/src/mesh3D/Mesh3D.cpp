#include "Mesh3D.hpp"

#include <igl/copyleft/tetgen/tetrahedralize.h>

namespace poly_fem
{
	void Mesh3D::refine(const int n_refiniment)
	{
//TODO implement me
	}

	bool Mesh3D::load(const std::string &path)
	{
		FILE *f = fopen(path.data(), "rt");
		if (!f) return false;

		int nv, np, nh;
		fscanf(f, "%d %d %d", &nv, &np, &nh);
		nh /= 3;

		mesh_.points.resize(3, nv);
		mesh_.vertices.resize(nv);

		for (int i = 0; i<nv; i++) {
			double x, y, z;
			fscanf(f, "%lf %lf %lf", &x, &y, &z);
			mesh_.points(0, i) = x;
			mesh_.points(1, i) = y;
			mesh_.points(2, i) = z;
			Vertex v;
			v.id = i;
			mesh_.vertices[i] = v;
		}
		mesh_.faces.resize(np);
		for (int i = 0; i<np; i++) {
			Face &p = mesh_.faces[i];
			p.id = i;
			int nw;

			fscanf(f, "%d", &nw);
			p.vs.resize(nw);
			for (int j = 0; j<nw; j++) {
				fscanf(f, "%d", &(p.vs[j]));
			}
		}
		mesh_.elements.resize(nh);
		for (int i = 0; i<nh; i++) {
			Element &h = mesh_.elements[i];
			h.id = i;

			int nf;
			fscanf(f, "%d", &nf);
			h.fs.resize(nf);

			for (int j = 0; j<nf; j++) {
				fscanf(f, "%d", &(h.fs[j]));
			}

			for (auto fid : h.fs)h.vs.insert(h.vs.end(), mesh_.faces[fid].vs.begin(), mesh_.faces[fid].vs.end());
				sort(h.vs.begin(), h.vs.end()); h.vs.erase(unique(h.vs.begin(), h.vs.end()), h.vs.end());

			int tmp; fscanf(f, "%d", &tmp);
			for (int j = 0; j<nf; j++) {
				int s;
				fscanf(f, "%d", &s);
				h.fs_flag.push_back(s);
			}
		}
		for (int i = 0; i<nh; i++) {
			int tmp;
			fscanf(f, "%d", &tmp);
			mesh_.elements[i].hex = tmp;
		}

		fclose(f);

		Navigation3D::prepare_mesh(mesh_);
		return true;
	}

	bool Mesh3D::save(const std::string &path) const{

		std::fstream f(path, std::ios::out);

		f << mesh_.points.cols() << " " << mesh_.faces.size() << " " << 3 * mesh_.elements.size() << std::endl;
		for (int i = 0; i<mesh_.points.cols(); i++)
			f << mesh_.points(0, i) << " " << mesh_.points(1, i) << " " << mesh_.points(2, i) << std::endl;

		for (auto f_ : mesh_.faces) {
			f << f_.vs.size() << " ";
			for (auto vid : f_.vs)
				f << vid << " ";
			f << std::endl;
		}

		for (uint32_t i = 0; i < mesh_.elements.size(); i++) {
			f << mesh_.elements[i].fs.size() << " ";
			for (auto fid : mesh_.elements[i].fs)
				f << fid << " ";
			f << std::endl;
			f << mesh_.elements[i].fs_flag.size() << " ";
			for (auto f_flag : mesh_.elements[i].fs_flag)
				f << f_flag << " ";
			f << std::endl;
		}

		for (uint32_t i = 0; i < mesh_.elements.size(); i++) {
			f << mesh_.elements[i].hex << std::endl;
		}

		f.close();

		return true;
	}


	double Mesh3D::compute_mesh_size() const
	{
//TODO implement me
		assert(false);
		return -1;
	}

	void Mesh3D::triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts) const
	{
		std::vector<Eigen::MatrixXi> local_tris(mesh_.elements.size());
		std::vector<Eigen::MatrixXd> local_pts(mesh_.elements.size());
		Eigen::MatrixXi tets;

		int total_tris = 0;
		int total_pts  = 0;

		for(std::size_t e = 0; e < mesh_.elements.size(); ++e)
		{
			const Element &el = mesh_.elements[e];

			const int n_vertices = el.vs.size();
			const int n_faces = el.fs.size();

			Eigen::MatrixXd local_pt(n_vertices, 3);
			Eigen::MatrixXi local_faces(2*n_faces, 3);

			std::map<int, int> global_to_local;

			for(int i = 0; i < n_vertices; ++i)
			{
				const int global_index = el.vs[i];
				local_pt.row(i) = mesh_.points.col(global_index).transpose();
				global_to_local[global_index] = i;
			}

			for(int i = 0; i < n_faces; ++i)
			{
				const Face &f = mesh_.faces[el.fs[i]];
				//TODO only quad faces;
				assert(f.vs.size() == 4);


				local_faces(2*i, 0) = global_to_local[f.vs[0]];
				local_faces(2*i, 1) = global_to_local[f.vs[1]];
				local_faces(2*i, 2) = global_to_local[f.vs[2]];

				local_faces(2*i+1, 0) = global_to_local[f.vs[0]];
				local_faces(2*i+1, 1) = global_to_local[f.vs[2]];
				local_faces(2*i+1, 2) = global_to_local[f.vs[3]];
			}

			igl::copyleft::tetgen::tetrahedralize(local_pt, local_faces, "QpYS0", local_pts[e], tets, local_tris[e]);

			total_tris += local_tris[e].rows();
			total_pts  += local_pts[e].rows();

			assert(local_pts[e].rows() == local_pt.rows());
		}


		tris.resize(total_tris, 3);
		pts.resize(total_pts, 3);

		int tri_index = 0;
		int pts_index = 0;
		for(std::size_t i = 0; i < local_tris.size(); ++i){
			tris.block(tri_index, 0, local_tris[i].rows(), local_tris[i].cols()) = local_tris[i].array() + pts_index;
			tri_index += local_tris[i].rows();

			pts.block(pts_index, 0, local_pts[i].rows(), local_pts[i].cols()) = local_pts[i];
			pts_index += local_pts[i].rows();
		}
	}

	void Mesh3D::set_boundary_tags(std::vector<int> &tags) const
	{
//TODO implement me
	}

	void Mesh3D::point(const int global_index, Eigen::MatrixXd &pt) const
	{
		pt = mesh_.points.col(global_index).transpose();
	}

	void Mesh3D::get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const
	{
		p0.resize(mesh_.edges.size(), 3);
		p1.resize(p0.rows(), p0.cols());

		Eigen::MatrixXd p0t, p1t;
		for(std::size_t e = 0; e < mesh_.edges.size(); ++e)
		{
			const int v0 = mesh_.edges[e].vs[0];
			const int v1 = mesh_.edges[e].vs[1];

			point(v0, p0t); point(v1, p1t);

			p0.row(e) = p0t;
			p1.row(e) = p1t;
		}
	}

//get nodes ids

	int Mesh3D::face_node_id(const int edge_id) const
	{
//TODO implement me
		assert(false);
		return -1;
	}

	int Mesh3D::edge_node_id(const int edge_id) const
	{
//TODO implement me
		assert(false);
		return -1;
	}

	int Mesh3D::vertex_node_id(const int vertex_id) const
	{
//TODO implement me
		assert(false);
		return -1;
	}

	bool Mesh3D::node_id_from_face_index(const Navigation3D::Index &index, int &id) const
	{
//TODO implement me
		assert(false);
		return -1;
	}


//get nodes positions

	Eigen::MatrixXd Mesh3D::node_from_element(const int el_id) const
	{
//TODO implement me
		assert(false);
		return Eigen::MatrixXd();
	}

	Eigen::MatrixXd Mesh3D::node_from_edge_index(const Navigation3D::Index &index) const
	{
//TODO implement me
		assert(false);
		return Eigen::MatrixXd();
	}

	Eigen::MatrixXd Mesh3D::node_from_face(const int face_id) const
	{
//TODO implement me
		assert(false);
		return Eigen::MatrixXd();
	}

	Eigen::MatrixXd Mesh3D::node_from_vertex(const int vertex_id) const
	{
//TODO implement me
		assert(false);
		return Eigen::MatrixXd();
	}

//navigation wrapper
	Navigation3D::Index Mesh3D::get_index_from_element_face(int hi, int lf, int lv) const
	{
		return Navigation3D::get_index_from_element_face(mesh_, hi, lf, lv);
	}

// Navigation in a surface mesh
	Navigation3D::Index Mesh3D::switch_vertex(Navigation3D::Index idx) const
	{
		return Navigation3D::switch_vertex(mesh_, idx);
	}

	Navigation3D::Index Mesh3D::switch_edge(Navigation3D::Index idx) const
	{
		return Navigation3D::switch_edge(mesh_, idx);
	}

	Navigation3D::Index Mesh3D::switch_face(Navigation3D::Index idx) const
	{
		return Navigation3D::switch_face(mesh_, idx);
	}

	Navigation3D::Index Mesh3D::switch_element(Navigation3D::Index idx) const
	{
		return Navigation3D::switch_element(mesh_, idx);
	}
}

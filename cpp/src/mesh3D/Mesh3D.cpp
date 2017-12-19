#include "Mesh3D.hpp"


namespace poly_fem
{
	void Mesh3D::refine(const int n_refiniment)
	{
		//TODO implement me
		assert(false);
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
	}

	void Mesh3D::triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts) const
	{
		//TODO implement me
		assert(false);
	}

	void Mesh3D::set_boundary_tags(std::vector<int> &tags) const
	{
		//TODO implement me
		assert(false);
	}

	void Mesh3D::point(const int global_index, Eigen::MatrixXd &pt) const
	{
		pt = mesh_.points.row(global_index);
	}

	void Mesh3D::get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1)
	{
		//TODO implement me
		assert(false);
	}

		//get nodes ids

	int Mesh3D::face_node_id(const int edge_id) const
	{
		//TODO implement me
		assert(false);
	}

	int Mesh3D::edge_node_id(const int edge_id) const
	{
		//TODO implement me
		assert(false);
	}

	int Mesh3D::vertex_node_id(const int vertex_id) const
	{
		//TODO implement me
		assert(false);
	}

	bool Mesh3D::node_id_from_face_index(const Navigation3D::Index &index, int &id) const
	{
		//TODO implement me
		assert(false);
	}


		//get nodes positions

	Eigen::MatrixXd Mesh3D::node_from_element(const int el_id) const
	{
		//TODO implement me
		assert(false);
	}

	Eigen::MatrixXd Mesh3D::node_from_edge_index(const Navigation3D::Index &index) const
	{
		//TODO implement me
		assert(false);
	}

	Eigen::MatrixXd Mesh3D::node_from_face(const int face_id) const
	{
		//TODO implement me
		assert(false);
	}
	
	Eigen::MatrixXd Mesh3D::node_from_vertex(const int vertex_id) const
	{
		//TODO implement me
		assert(false);
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

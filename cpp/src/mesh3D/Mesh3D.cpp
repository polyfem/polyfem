#include "Mesh3D.hpp"

#include <fstream>
#include <igl/copyleft/tetgen/tetrahedralize.h>

namespace poly_fem
{
	void Mesh3D::refine(const int n_refiniment, const double t, std::vector<int> &parent_nodes)
	{
		//TODO to aware refiniement

		MeshProcessing3D::refine_catmul_clark_polar(mesh_, n_refiniment, parent_nodes);
		Navigation3D::prepare_mesh(mesh_);

		compute_elements_tag();
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

		char s[1024], sread[1024];
		int find = false, num = 0;
		while (!feof(f) && !find) {
			fgets(s, 1024, f);
			if (sscanf(s, "%s%d", &sread, &num) == 2 && (strcmp(sread, "KERNEL") == 0)) find = true;
		}
		if (find) {
			for (int i = 0; i<nh; i++) {
				double x, y, z;
				fscanf(f, "%lf %lf %lf", &x, &y, &z);
				mesh_.elements[i].v_in_Kernel.push_back(x);
				mesh_.elements[i].v_in_Kernel.push_back(y);
				mesh_.elements[i].v_in_Kernel.push_back(z);
			}
		}

		fclose(f);

		auto &V = mesh_.points;
		V = (V.colwise() - V.rowwise().minCoeff()) / (V.rowwise().maxCoeff() - V.rowwise().minCoeff()).maxCoeff();

		//TODO not so nice to detect triangle meshes
		is_simplicial_ = n_cell_vertices(0) == 4;

		if(is_simplicial_)
		{
			for(int i = 0; i < n_cells(); ++i)
			{
				assert(n_cell_vertices(i) == 4);
				std::array<GEO::vec3, 4> vertices;
				auto &face_vertices = mesh_.elements[i].vs;

				for(int lv = 0; lv < 4; ++lv)
				{
					auto pt = point(face_vertices[lv]);
					for(int d = 0; d < 3; ++d)
					{
						vertices[lv][d] = pt(d);
					}
				}

				const double vol = GEO::Geom::tetra_signed_volume(vertices[0], vertices[1], vertices[2], vertices[4]);
				if(vol < 0)
				{
					std::swap(face_vertices[1], face_vertices[2]);
				}
			}
		}

		Navigation3D::prepare_mesh(mesh_);
		// if(is_simplicial())
			MeshProcessing3D::orient_volume_mesh(mesh_);
		compute_elements_tag();
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

		f << "KERNEL" << " " << mesh_.elements.size() << std::endl;
		for (uint32_t i = 0; i < mesh_.elements.size(); i++) {
			f << mesh_.elements[i].v_in_Kernel[0] <<" " << mesh_.elements[i].v_in_Kernel[1] << " " << mesh_.elements[i].v_in_Kernel[2]<< std::endl;
		}
		f.close();

		return true;
	}

	bool Mesh3D::save(const std::vector<int> &eles, const int ringN, const std::string &path) const {
		Mesh3DStorage mesh = mesh_;
		mesh.edges.clear();
		mesh.faces.clear();
		mesh.elements.clear();

		std::vector<bool> H_flag(mesh_.elements.size(), false);
		for (auto i : eles)H_flag[i] = true;

		for (int i = 0; i < ringN; i++) {
			std::vector<bool> H_flag_(H_flag.size(), false);
			for (uint32_t j = 0; j < H_flag.size(); j++) if (H_flag[j]) {
				for (const auto vid : mesh_.elements[j].vs)for (const auto nhid : mesh_.vertices[vid].neighbor_hs)H_flag_[nhid] = true;
			}
			H_flag = H_flag_;
		}

		std::vector<bool> F_flag(mesh_.faces.size(), false);
		for (int i = 0; i < H_flag.size();i++)if (H_flag[i]) {
			for(auto fid:mesh_.elements[i].fs) F_flag[fid] = true;
		}

		std::vector<int32_t> F_map(mesh_.faces.size(), -1), F_map_reverse;
		for (auto f : mesh_.faces)if (F_flag[f.id]) {
			Face f_;
			f_.id = mesh.faces.size();
			f_.vs = f.vs;
			F_map[f.id] = f_.id;
			F_map_reverse.push_back(f.id);

			mesh.faces.push_back(f_);
		}

		for (auto ele : mesh_.elements)if (H_flag[ele.id]) {
			Element ele_;
			for (auto fid : ele.fs)ele_.fs.push_back(F_map[fid]);
			ele_.fs_flag = ele.fs_flag;
			ele_.hex = ele.hex;
			ele_.v_in_Kernel = ele.v_in_Kernel;
			ele_.vs = ele.vs;

			mesh.elements.push_back(ele_);
		}

		//save
		std::fstream f(path, std::ios::out);

		f << mesh.points.cols() << " " << mesh.faces.size() << " " << 3 * mesh.elements.size() << std::endl;
		for (int i = 0; i<mesh_.points.cols(); i++)
			f << mesh.points(0, i) << " " << mesh.points(1, i) << " " << mesh.points(2, i) << std::endl;

		for (auto f_ : mesh.faces) {
			f << f_.vs.size() << " ";
			for (auto vid : f_.vs)
				f << vid << " ";
			f << std::endl;
		}

		for (uint32_t i = 0; i < mesh.elements.size(); i++) {
			f << mesh.elements[i].fs.size() << " ";
			for (auto fid : mesh.elements[i].fs)
				f << fid << " ";
			f << std::endl;
			f << mesh.elements[i].fs_flag.size() << " ";
			for (auto f_flag : mesh.elements[i].fs_flag)
				f << f_flag << " ";
			f << std::endl;
		}

		for (uint32_t i = 0; i < mesh.elements.size(); i++) {
			f << mesh.elements[i].hex << std::endl;
		}

		f << "KERNEL" << " " << mesh.elements.size() << std::endl;
		for (uint32_t i = 0; i < mesh.elements.size(); i++) {
			f << mesh.elements[i].v_in_Kernel[0] << " " << mesh.elements[i].v_in_Kernel[1] << " " << mesh.elements[i].v_in_Kernel[2] << std::endl;
		}
		f.close();

		return true;
	}

	void Mesh3D::triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const
	{
		ranges.clear();

		std::vector<Eigen::MatrixXi> local_tris(mesh_.elements.size());
		std::vector<Eigen::MatrixXd> local_pts(mesh_.elements.size());
		Eigen::MatrixXi tets;

		int total_tris = 0;
		int total_pts  = 0;

		ranges.push_back(0);

		Eigen::MatrixXd face_barys;
		face_barycenters(face_barys);

		Eigen::MatrixXd cell_barys;
		cell_barycenters(cell_barys);

		for(std::size_t e = 0; e < mesh_.elements.size(); ++e)
		{
			const Element &el = mesh_.elements[e];

			const int n_vertices = el.vs.size();
			const int n_faces = el.fs.size();

			Eigen::MatrixXd local_pt(n_vertices+n_faces, 3);

			std::map<int, int> global_to_local;

			for(int i = 0; i < n_vertices; ++i)
			{
				const int global_index = el.vs[i];
				local_pt.row(i) = mesh_.points.col(global_index).transpose();
				global_to_local[global_index] = i;
			}

			int n_local_faces = 0;
			for(int i = 0; i < n_faces; ++i)
			{
				const Face &f = mesh_.faces[el.fs[i]];
				n_local_faces += f.vs.size();

				local_pt.row(n_vertices+i) = face_barys.row(f.id); // node_from_face(f.id);
			}


			Eigen::MatrixXi local_faces(n_local_faces, 3);

			int face_index = 0;
			for(int i = 0; i < n_faces; ++i)
			{
				const Face &f = mesh_.faces[el.fs[i]];
				const int n_face_vertices = f.vs.size();

				const Eigen::RowVector3d e0 = (point(f.vs[0]) - local_pt.row(n_vertices+i));
				const Eigen::RowVector3d e1 = (point(f.vs[1]) - local_pt.row(n_vertices+i));
				const Eigen::RowVector3d normal = e0.cross(e1);
				// const Eigen::RowVector3d check_dir = (node_from_element(e)-p);
				const Eigen::RowVector3d check_dir = (cell_barys.row(e)-point(f.vs[1]));

				const bool reverse_order = normal.dot(check_dir) > 0;

				for(int j = 0; j < n_face_vertices; ++j)
				{
					const int jp = (j + 1) % n_face_vertices;
					if(reverse_order)
					{
						local_faces(face_index, 0) = global_to_local[f.vs[jp]];
						local_faces(face_index, 1) = global_to_local[f.vs[j]];
					}
					else
					{
						local_faces(face_index, 0) = global_to_local[f.vs[j]];
						local_faces(face_index, 1) = global_to_local[f.vs[jp]];
					}
					local_faces(face_index, 2) = n_vertices + i;

					++face_index;
				}
			}

			local_pts[e] = local_pt;
			local_tris[e] = local_faces;
			// igl::copyleft::tetgen::tetrahedralize(local_pt, local_faces, "QpYS0", local_pts[e], tets, local_tris[e]);

			total_tris += local_tris[e].rows();
			total_pts  += local_pts[e].rows();

			ranges.push_back(total_tris);

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

	void Mesh3D::fill_boundary_tags(std::vector<int> &tags) const
	{
		//TODO implement me
		// tags.resize(mesh_.faces.size());
		// std::fill(tags.begin(), tags.end(), -1);

		// for(std::size_t f = 0; f < mesh_.faces.size(); ++f)
		// {
		// 	if(!mesh_.faces[f].boundary)
		// 		continue;

		// 	const auto p = node_from_face(f);


		// 	if(fabs(p(0))<1e-8)
		// 		tags[f]=1;
		// 	if(fabs(p(1))<1e-8)
		// 		tags[f]=2;
		// 	if(fabs(p(2))<1e-8)
		// 		tags[f]=5;
		// 	if(fabs(p(0)-1)<1e-8)
		// 		tags[f]=3;
		// 	if(fabs(p(1)-1)<1e-8)
		// 		tags[f]=4;
		// 	if(fabs(p(2)-1)<1e-8)
		// 		tags[f]=6;
		// }
	}

	RowVectorNd Mesh3D::point(const int global_index) const {
		RowVectorNd pt = mesh_.points.col(global_index).transpose();
		return pt;
	}

	void Mesh3D::get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const
	{
		p0.resize(mesh_.edges.size(), 3);
		p1.resize(p0.rows(), p0.cols());

		for(std::size_t e = 0; e < mesh_.edges.size(); ++e)
		{
			const int v0 = mesh_.edges[e].vs[0];
			const int v1 = mesh_.edges[e].vs[1];

			p0.row(e) = point(v0);
			p1.row(e) = point(v1);
		}
	}

	// int Mesh3D::face_node_id(const int face_id) const
	// {
	// 	return faces_node_id_[face_id];
	// }

	// int Mesh3D::edge_node_id(const int edge_id) const
	// {
	// 	return edges_node_id_[edge_id];
	// }

	// int Mesh3D::vertex_node_id(const int vertex_id) const
	// {
	// 	return vertices_node_id_[vertex_id];
	// }

	// bool Mesh3D::node_id_from_face_index(const Navigation3D::Index &index, int &id) const
	// {
	// 	id = switch_element(index).element;
	// 	bool is_real_boundary = true;
	// 	if(id >= 0)
	// 	{
	// 		is_real_boundary = false;
	// 		if(n_element_vertices(id) == 8 && n_element_faces(id) == 6)
	// 			return is_real_boundary;

	// 	}

	// 	id = face_node_id(index.face);
	// 	assert(id >= 0);

	// 	return is_real_boundary;
	// }

	// bool Mesh3D::node_id_from_edge_index(const Navigation3D::Index &index, int &id) const
	// {
	// 	Navigation3D::Index new_index = switch_element(index);
	// 	id = new_index.element;

	// 	auto is_polyhedron = [this](int e) {
	// 		return (n_element_vertices(e) != 8) || (n_element_faces(e) != 6);
	// 	};

	// 	if(id < 0 || is_polyhedron(id))
	// 	{
	// 		new_index = switch_element(switch_face(index));
	// 		id = new_index.element;

	// 		if(id < 0 || is_polyhedron(id))
	// 		{
	// 			const bool is_boundary = id < 0;
	// 			id = edge_node_id(index.edge);
	// 			return is_boundary;
	// 		}

	// 		return node_id_from_face_index(switch_face(new_index), id);
	// 	}

	// 	return node_id_from_face_index(switch_face(new_index), id);
	// }


	// int Mesh3D::node_id_from_vertex_index_explore(const Navigation3D::Index &index, int &id, Eigen::MatrixXd &node, bool &real_b) const
	// {
	// 	auto is_polyhedron = [this](int e) {
	// 		return (n_element_vertices(e) != 8) || (n_element_faces(e) != 6);
	// 	};

	// 	Navigation3D::Index new_index = switch_element(index);

	// 	id = new_index.element;
	// 	real_b = id < 0;

	// 	if(id < 0 || is_polyhedron(id))
	// 	{
	// 		id = vertex_node_id(index.vertex);
	// 		node = node_from_vertex(index.vertex);
	// 		return 3;
	// 	}

	// 	new_index = switch_element(switch_face(new_index));
	// 	id = new_index.element;
	// 	real_b = id < 0;

	// 	if(id < 0 || is_polyhedron(id))
	// 	{
	// 		id = edge_node_id(switch_edge(new_index).edge);
	// 		node = node_from_edge(switch_edge(new_index).edge);
	// 		return 2;
	// 	}

	// 	new_index = switch_element(switch_face(switch_edge(new_index)));
	// 	id = new_index.element;
	// 	real_b = id < 0;

	// 	if(id < 0 || is_polyhedron(id))
	// 	{
	// 		id = face_node_id(new_index.face);
	// 		node = node_from_face(new_index.face);
	// 		return 1;
	// 	}

	// 	node = node_from_element(id);
	// 	return 0;
	// }

	// bool Mesh3D::node_id_from_vertex_index(const Navigation3D::Index &index, int &id) const
	// {
	// 	std::array<int, 6> path;
	// 	std::array<int, 6> ids;
	// 	std::array<bool, 6> real_b;
	// 	Eigen::MatrixXd node;

	// 	path[0] = node_id_from_vertex_index_explore(index, ids[0], node, real_b[0]);
	// 	path[1] = node_id_from_vertex_index_explore(switch_face(index), ids[1], node, real_b[1]);

	// 	path[2] = node_id_from_vertex_index_explore(switch_edge(index), ids[2], node, real_b[2]);
	// 	path[3] = node_id_from_vertex_index_explore(switch_face(switch_edge(index)), ids[3], node, real_b[3]);

	// 	path[4] = node_id_from_vertex_index_explore(switch_edge(switch_face(index)), ids[4], node, real_b[4]);
	// 	path[5] = node_id_from_vertex_index_explore(switch_face(switch_edge(switch_face(index))), ids[5], node, real_b[5]);

	// 	const int min_path = *std::min_element(path.begin(), path.end());

	// 	bool res = min_path > 0;
	// 	for(int i = 0 ; i < 6; ++i)
	// 	{
	// 		if(path[i]==min_path)
	// 		{
	// 			id = ids[i];
	// 			res = real_b[i];
	// 			break;
	// 		}
	// 	}

	// 	return res;
	// }

	// Eigen::MatrixXd Mesh3D::node_from_edge_index(const Navigation3D::Index &index) const
	// {
	// 	Navigation3D::Index new_index = switch_element(index);
	// 	int id = new_index.element;

	// 	auto is_polyhedron = [this](int e) {
	// 		return (n_element_vertices(e) != 8) || (n_element_faces(e) != 6);
	// 	};

	// 	if(id < 0 || is_polyhedron(id))
	// 	{
	// 		new_index = switch_element(switch_face(index));
	// 		id = new_index.element;
	// 		if(id < 0 || is_polyhedron(id))
	// 			return node_from_edge(index.edge);

	// 		return node_from_face_index(switch_face(new_index));
	// 	}

	// 	return node_from_face_index(switch_face(new_index));
	// }

	// Eigen::MatrixXd Mesh3D::node_from_vertex_index(const Navigation3D::Index &index) const
	// {
	// 	std::array<int, 6> path;
	// 	std::array<Eigen::MatrixXd, 6> nodes;
	// 	int id;
	// 	bool real_b;

	// 	path[0] = node_id_from_vertex_index_explore(index, id, nodes[0], real_b);
	// 	path[1] = node_id_from_vertex_index_explore(switch_face(index), id, nodes[1], real_b);

	// 	path[2] = node_id_from_vertex_index_explore(switch_edge(index), id, nodes[2], real_b);
	// 	path[3] = node_id_from_vertex_index_explore(switch_face(switch_edge(index)), id, nodes[3], real_b);

	// 	path[4] = node_id_from_vertex_index_explore(switch_edge(switch_face(index)), id, nodes[4], real_b);
	// 	path[5] = node_id_from_vertex_index_explore(switch_face(switch_edge(switch_face(index))), id, nodes[5], real_b);

	// 	const int min_path = *std::min_element(path.begin(), path.end());

	// 	for(int i = 0 ; i < 6; ++i)
	// 	{
	// 		if(path[i]==min_path)
	// 		{
	// 			return nodes[i];
	// 		}
	// 	}

	// 	assert(false);
	// 	return nodes[0];
	// }


	// Eigen::MatrixXd Mesh3D::node_from_element(const int el_id) const
	// {
	// 	//warning, if change my have side effects
	// 	Eigen::MatrixXd res(1,3), p;
	// 	res.setZero();

	// 	for(std::size_t j = 0; j < mesh_.elements[el_id].vs.size(); ++j)
	// 	{
	// 		point(mesh_.elements[el_id].vs[j], p);
	// 		res += p;
	// 	}

	// 	res /= mesh_.elements[el_id].vs.size();

	// 	return res;
	// }

	// Eigen::MatrixXd Mesh3D::node_from_face_index(const Navigation3D::Index &index) const
	// {
	// 	int id = switch_element(index).element;
	// 	if(id >= 0)
	// 	{
	// 		if(n_element_vertices(id) == 8 && n_element_faces(id) == 6)
	// 			return node_from_element(id);
	// 	}

	// 	id = face_node_id(index.face);
	// 	assert(id >= 0);

	// 	return faces_node_[index.face];
	// }

	// Eigen::MatrixXd Mesh3D::node_from_face(const int face_id) const
	// {
	// 	return faces_node_[face_id];
	// }

	// Eigen::MatrixXd Mesh3D::node_from_edge(const int edge_id) const
	// {
	// 	return edges_node_[edge_id];
	// }

	// Eigen::MatrixXd Mesh3D::node_from_vertex(const int vertex_id) const
	// {
	// 	return vertices_node_[vertex_id];
	// }


	// bool Mesh3D::is_boundary_edge(int eid) {
	// 	return mesh_.edges[eid].boundary_hex;
	// }
	// bool Mesh3D::is_boundary_vertex(int vid) {
	// 	return mesh_.vertices[vid].boundary_hex;
	// }


	void Mesh3D::compute_elements_tag()
	{
		std::vector<ElementType> &ele_tag = elements_tag_;
		ele_tag.clear();

		ele_tag.resize(mesh_.elements.size());
		for (auto &t : ele_tag) t = ElementType::RegularInteriorCube;

		//boundary flags
		std::vector<bool> bv_flag(mesh_.vertices.size(), false), be_flag(mesh_.edges.size(), false), bf_flag(mesh_.faces.size(), false);
		for (auto f : mesh_.faces)if (f.boundary)bf_flag[f.id] = true;
		else {
			for(auto nhid:f.neighbor_hs)if(!mesh_.elements[nhid].hex)bf_flag[f.id] = true;
		}
		for (uint32_t i = 0; i < mesh_.faces.size(); ++i)
			if (bf_flag[i]) for (uint32_t j = 0; j < mesh_.faces[i].vs.size(); ++j) {
				uint32_t eid = mesh_.faces[i].es[j];
				be_flag[eid] = true;
				bv_flag[mesh_.faces[i].vs[j]] = true;
			}

		for (auto &ele:mesh_.elements) {
			if (ele.hex) {
				bool attaching_non_hex = false, on_boundary = false;;
				for (auto vid : ele.vs){
					for (auto eleid : mesh_.vertices[vid].neighbor_hs) if (!mesh_.elements[eleid].hex) {
						attaching_non_hex = true; break;
					}
					if (mesh_.vertices[vid].boundary) {
						on_boundary = true; break;
					}
					if (on_boundary || attaching_non_hex) break;
				}
				if (on_boundary || attaching_non_hex) {
					ele_tag[ele.id] = ElementType::MultiSingularBoundaryCube;
					//has no boundary edge--> singular
					bool boundary_edge = false, boundary_edge_singular = false, interior_edge_singular = false;
					int n_interior_edge_singular = 0;
					for (auto eid : ele.es) {
						int en = 0;
						if (be_flag[eid]) {
							boundary_edge = true;
							for (auto nhid : mesh_.edges[eid].neighbor_hs)if (mesh_.elements[nhid].hex)en++;
							if (en > 2)boundary_edge_singular = true;
						}
						else {
							for (auto nhid : mesh_.edges[eid].neighbor_hs)if (mesh_.elements[nhid].hex)en++;
							if (en != 4) {
								interior_edge_singular = true; n_interior_edge_singular++;
							}
						}
					}
					if (!boundary_edge || boundary_edge_singular || n_interior_edge_singular > 1)continue;

					bool has_singular_v = false, has_iregular_v = false; int n_in_irregular_v = 0;
					for (auto vid : ele.vs) {
						int vn = 0;
						if (bv_flag[vid]) {
							int nh = 0;
							for (auto nhid : mesh_.vertices[vid].neighbor_hs)if (mesh_.elements[nhid].hex)nh++;
							if (nh > 4)has_iregular_v = true;
							continue;//not sure the conditions
						}
						else {
							if (mesh_.vertices[vid].neighbor_hs.size() != 8)n_in_irregular_v++;
							int n_irregular_e = 0;
							for (auto eid : mesh_.vertices[vid].neighbor_es) {
								if (mesh_.edges[eid].neighbor_hs.size() != 4)
									n_irregular_e++;
							}
							if (n_irregular_e != 0 && n_irregular_e != 2) {
								has_singular_v = true; break;
							}
						}
					}
					int n_irregular_e = 0;
					for (auto eid : ele.es) if (!be_flag[eid] && mesh_.edges[eid].neighbor_hs.size() != 4)
						n_irregular_e++;
					if (has_singular_v) continue;
					if (!has_singular_v) {
						if (n_irregular_e == 1) ele_tag[ele.id] = ElementType::SimpleSingularBoundaryCube;
						else if (n_irregular_e == 0 && n_in_irregular_v == 0 && !has_iregular_v) ele_tag[ele.id] = ElementType::RegularBoundaryCube;
						else continue;
					}
					continue;
				}

			//type 1
				bool has_irregular_v = false;
				for (auto vid : ele.vs)  if (mesh_.vertices[vid].neighbor_hs.size() != 8) {
					has_irregular_v = true; break;
				}
				if(!has_irregular_v){
					ele_tag[ele.id] = ElementType::RegularInteriorCube;
					continue;
				}
			//type 2
				bool has_singular_v = false; int n_irregular_v = 0;
				for (auto vid : ele.vs){
					if (mesh_.vertices[vid].neighbor_hs.size() != 8)
						n_irregular_v++;
					int n_irregular_e = 0;
					for (auto eid : mesh_.vertices[vid].neighbor_es){
						if (mesh_.edges[eid].neighbor_hs.size() != 4)
							n_irregular_e++;
					}
					if (n_irregular_e!=0 && n_irregular_e != 2) {
						has_singular_v = true; break;
					}
				}
				if (!has_singular_v && n_irregular_v == 2) {
					ele_tag[ele.id] = ElementType::SimpleSingularInteriorCube;
					continue;
				}

				ele_tag[ele.id] = ElementType::MultiSingularInteriorCube;
			}
			else {
				ele_tag[ele.id] = ElementType::InteriorPolytope;
				for (auto fid : ele.fs)if (mesh_.faces[fid].boundary) { ele_tag[ele.id] = ElementType::BoundaryPolytope; break; }
			}
		}
	}

	RowVectorNd Mesh3D::edge_barycenter(const int e) const {
		const int v0 = mesh_.edges[e].vs[0];
		const int v1 = mesh_.edges[e].vs[1];
		return 0.5*(point(v0) + point(v1));
	}

	RowVectorNd Mesh3D::face_barycenter(const int f) const {
		const int n_vertices = n_face_vertices(f);
		RowVectorNd bary(3); bary.setZero();

		const auto &vertices = mesh_.faces[f].vs;
		for(int lv = 0; lv < n_vertices; ++lv) {
			bary += point(vertices[lv]);
		}
		return bary / n_vertices;
	}

	RowVectorNd Mesh3D::cell_barycenter(const int c) const {
		const int n_vertices = n_cell_vertices(c);
		RowVectorNd bary(3); bary.setZero();

		const auto &vertices = mesh_.elements[c].vs;
		for(int lv = 0; lv < n_vertices; ++lv)
		{
			bary += point(vertices[lv]);
		}
		return bary / n_vertices;
	}

	void Mesh3D::to_face_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> &to_face) const
	{
        //top
		to_face[0]= [this](Navigation3D::Index idx) { return switch_face(switch_edge(switch_vertex(switch_edge(switch_face(idx))))); };
        //bottom
		to_face[1]= [this](Navigation3D::Index idx) { return idx; };

        //left
		to_face[2]= [this](Navigation3D::Index idx) { return switch_face(switch_edge(switch_vertex(idx))); };
        //right
		to_face[3]= [this](Navigation3D::Index idx) { return switch_face(switch_edge(idx)); };

        //back
		to_face[4]= [this](Navigation3D::Index idx) { return switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(idx))))); };
        //front
		to_face[5]= [this](Navigation3D::Index idx) { return switch_face(idx); };
	}

	void Mesh3D::to_vertex_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 8> &to_vertex) const
	{
		to_vertex[0]= [this](Navigation3D::Index idx) { return idx; };
		to_vertex[1]= [this](Navigation3D::Index idx) { return switch_vertex(idx); };
		to_vertex[2]= [this](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_vertex(idx))); };
		to_vertex[3]= [this](Navigation3D::Index idx) { return switch_vertex(switch_edge(idx)); };

		to_vertex[4]= [this](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_face(idx))); };
		to_vertex[5]= [this](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_vertex(switch_edge(switch_face(idx))))); };
		to_vertex[6]= [this](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_face(switch_vertex(switch_edge(switch_vertex(idx)))))); };
		to_vertex[7]= [this](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_face(switch_vertex(switch_edge(idx))))); };
	}

	void Mesh3D::to_edge_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 12> &to_edge) const
	{
		to_edge[0]= [this](Navigation3D::Index idx) { return idx; };
		to_edge[1]= [this](Navigation3D::Index idx) { return switch_edge(switch_vertex(idx)); };
		to_edge[2]= [this](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))); };
		to_edge[3]= [this](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))); };

		to_edge[4]= [this](Navigation3D::Index idx) { return switch_edge(switch_face(idx)); };
		to_edge[5]= [this](Navigation3D::Index idx) { return switch_edge(switch_face(switch_edge(switch_vertex(idx)))); };
		to_edge[6]= [this](Navigation3D::Index idx) { return switch_edge(switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))); };
		to_edge[7]= [this](Navigation3D::Index idx) { return switch_edge(switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))))); };

		to_edge[8]= [this](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_face(idx)))); };
		to_edge[9]= [this](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_face(switch_edge(switch_vertex(idx)))))); };
		to_edge[10]= [this](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))))); };
		to_edge[11]= [this](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))))))); };
	}

	//   v7────v6
	//   ╱┆    ╱│
	// v4─┼──v5 │
	//  │v3┄┄┄┼v2
	//  │╱    │╱
	// v0────v1
	std::array<int, 8> Mesh3D::get_ordered_vertices_from_hex(const int element_index) const {
		assert(is_cube(element_index));
		auto idx = get_index_from_element(element_index);
		std::array<int, 8> v;

        std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 8> to_vertex;
        to_vertex_functions(to_vertex);
        for(int i=0;i<8;++i)
        	v[i] = to_vertex[i](idx).vertex;


		// for (int lv = 0; lv < 4; ++lv) {
		// 	v[lv] = idx.vertex;
		// 	idx = next_around_face_of_element(idx);
		// }
		// // assert(idx == get_index_from_element(element_index));
		// idx = switch_face(switch_edge(switch_vertex(switch_edge(switch_face(idx)))));
		// for (int lv = 0; lv < 4; ++lv) {
		// 	v[4+lv] = idx.vertex;
		// 	idx = next_around_face_of_element(idx);
		// }
		return v;
	}

	std::array<int, 4> Mesh3D::get_ordered_vertices_from_tet(const int element_index) const
	{
		auto idx = get_index_from_element(element_index);
		std::array<int, 4> v;

       for (int lv = 0; lv < 3; ++lv) {
			v[lv] = idx.vertex;
			idx = next_around_face(idx);
		}
		// assert(idx == get_index_from_element(element_index));
		idx = switch_vertex(switch_edge(switch_face(idx)));
		v[3] = idx.vertex;


		std::array<GEO::vec3, 4> vertices;

		for(int lv = 0; lv < 4; ++lv)
		{
			auto pt = point(v[lv]);
			for(int d = 0; d < 3; ++d)
			{
				vertices[lv][d] = pt(d);
			}
		}

		const double vol = GEO::Geom::tetra_signed_volume(vertices[0], vertices[1], vertices[2], vertices[4]);
		std::cout<<vol<<std::endl;


		return v;
	}

	// void Mesh3D::create_boundary_nodes()
	// {
	// 	faces_node_id_.resize(mesh_.faces.size());
	// 	faces_node_.resize(mesh_.faces.size());

	// 	int counter = n_elements();

	// 	Eigen::Matrix<double, 1, 3> bary;
	// 	Eigen::MatrixXd p, p0, p1;

	// 	for (int f = 0; f < (int) mesh_.faces.size(); ++f)
	// 	{
	// 		const Face &face = mesh_.faces[f];
	// 		if(!face.boundary_hex)
	// 			faces_node_id_[f] = -1;
	// 		else
	// 			faces_node_id_[f] = counter++;

	// 		bary.setZero();

	// 		for(std::size_t i = 0; i < face.vs.size(); ++i){
	// 			point(face.vs[i], p);
	// 			bary += p;
	// 		}

	// 		bary /= face.vs.size();
	// 		faces_node_[f]=bary;
	// 	}

	// 	edges_node_id_.resize(mesh_.edges.size());
	// 	edges_node_.resize(mesh_.edges.size());

	// 	for (int e = 0; e < (int) mesh_.edges.size(); ++e)
	// 	{
	// 		const Edge &edge = mesh_.edges[e];
	// 		if(!edge.boundary_hex)
	// 			edges_node_id_[e] = -1;
	// 		else
	// 			edges_node_id_[e] = counter++;

	// 		const int v0 = edge.vs[0];
	// 		const int v1 = edge.vs[1];
	// 		point(v0, p0); point(v1, p1);
	// 		edges_node_[e]=(p0 + p1)/2;
	// 	}

	// 	vertices_node_id_.resize(mesh_.vertices.size());
	// 	vertices_node_.resize(mesh_.vertices.size());

	// 	for (int v = 0; v < (int) mesh_.vertices.size(); ++v)
	// 	{
	// 		const Vertex &vertex = mesh_.vertices[v];
	// 		if(!vertex.boundary_hex)
	// 			vertices_node_id_[v] = -1;
	// 		else
	// 			vertices_node_id_[v] = counter++;

	// 		point(vertex.id, p);
	// 		vertices_node_[v] = p;
	// 	}
	// }










	void Mesh3D::geomesh_2_mesh_storage(const GEO::Mesh &gm, Mesh3DStorage &m) {
		m.vertices.clear(); m.edges.clear(); m.faces.clear();
		m.vertices.resize(gm.vertices.nb());
		m.faces.resize(gm.facets.nb());
		for (uint32_t i = 0; i < m.vertices.size(); i++) {
			Vertex v;
			v.id = i;
			v.v.push_back(gm.vertices.point_ptr(i)[0]);
			v.v.push_back(gm.vertices.point_ptr(i)[1]);
			v.v.push_back(gm.vertices.point_ptr(i)[2]);
			m.vertices[i] = v;
		}
		m.points.resize(3, m.vertices.size());
		for (uint32_t i = 0; i < m.vertices.size(); i++) {
			m.points(0, i) = m.vertices[i].v[0];
			m.points(1, i) = m.vertices[i].v[1];
			m.points(2, i) = m.vertices[i].v[2];
		}

		if (m.type == MeshType::Tri || m.type == MeshType::Qua || m.type == MeshType::HSur) {
			for (uint32_t i = 0; i < m.faces.size(); i++) {
				Face f;
				f.id = i;
				f.vs.resize(gm.facets.nb_vertices(i));
				for (uint32_t j = 0; j < f.vs.size(); j++) {
					f.vs[j] = gm.facets.vertex(i, j);
				}
				m.faces[i] = f;
			}
			MeshProcessing3D::build_connectivity(m);
		}
	}
}

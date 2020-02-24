#include <polyfem/Mesh3D.hpp>
#include <polyfem/MeshUtils.hpp>
#include <polyfem/StringUtils.hpp>

#include <polyfem/Logger.hpp>

#include <geogram/mesh/mesh_io.h>
#include <fstream>

namespace polyfem
{
	void Mesh3D::refine(const int n_refiniment, const double t, std::vector<int> &parent_nodes)
	{
		if (n_refiniment <= 0)
		{
			return;
		}

		//TODO refine high order mesh!
		orders_.resize(0,0);
		if (mesh_.type == MeshType::Tet) {
			MeshProcessing3D::refine_red_refinement_tet(mesh_, n_refiniment);
		}
		else {
			for (size_t i = 0; i < elements_tag().size(); ++i)
			{
				if (elements_tag()[i] == ElementType::InteriorPolytope || elements_tag()[i] == ElementType::BoundaryPolytope)
					mesh_.elements[i].hex = false;
			}

			bool reverse_grow = false;
			MeshProcessing3D::refine_catmul_clark_polar(mesh_, n_refiniment, reverse_grow, parent_nodes);
		}

		Navigation3D::prepare_mesh(mesh_);
		compute_elements_tag();
	}

	bool Mesh3D::load(const std::string &path)
	{
		edge_nodes_.clear();
		face_nodes_.clear();
		cell_nodes_.clear();

		if (!StringUtils::endswidth(path, ".HYBRID")) {
			GEO::Mesh M;
			GEO::mesh_load(path, M);
			return load(M);
		}

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

		//remove horrible kernels and replace with barycenters
		for(int c = 0; c < n_cells(); ++c)
		{
			auto bary = cell_barycenter(c);
			for(int d  = 0; d < 3; ++d)
				mesh_.elements[c].v_in_Kernel[d] = bary(d);
		}

		Navigation3D::prepare_mesh(mesh_);
		// if(is_simplicial())
			// MeshProcessing3D::orient_volume_mesh(mesh_);
		compute_elements_tag();
		return true;
	}

	// load from a geogram surface mesh (for debugging), or volume mesh
	// if loading a surface mesh, it assumes there is only one polyhedral cell, and the last vertex id a point in the kernel
	bool Mesh3D::load(const GEO::Mesh &M) {
		edge_nodes_.clear();
		face_nodes_.clear();
		cell_nodes_.clear();

		assert(M.vertices.dimension() == 3);

		// Set vertices
		const int nv = M.vertices.nb();
		mesh_.points.resize(3, nv);
		mesh_.vertices.resize(nv);
		for (int i = 0; i < nv; ++i) {
			mesh_.points(0, i) = M.vertices.point(i)[0];
			mesh_.points(1, i) = M.vertices.point(i)[1];
			mesh_.points(2, i) = M.vertices.point(i)[2];
			Vertex v;
			v.id = i;
			mesh_.vertices[i] = v;
		}

		// Set cells
		if (M.cells.nb() == 0) {

			bool last_isolated = true;

			// Set faces
			mesh_.faces.resize(M.facets.nb());
			for (int i = 0; i < (int) M.facets.nb(); ++i) {
				Face &face = mesh_.faces[i];
				face.id = i;

				face.vs.resize(M.facets.nb_vertices(i));
				for (int j = 0; j < (int) M.facets.nb_vertices(i); ++j) {
					face.vs[j] = M.facets.vertex(i, j);
					if ((int) face.vs[j] == nv - 1) {
						last_isolated = false;
					}
				}
			}

			// Assumes there is only 1 polyhedron described by a closed input surface
			mesh_.elements.resize(1);
			for (int i = 0; i < 1; ++i) {
				Element &cell = mesh_.elements[i];
				cell.id = i;

				int nf = M.facets.nb();
				cell.fs.resize(nf);

				for (int j = 0; j < nf; ++j) {
					cell.fs[j] = j;
				}

				for (auto fid : cell.fs) {
					cell.vs.insert(cell.vs.end(), mesh_.faces[fid].vs.begin(), mesh_.faces[fid].vs.end());
				}
				sort(cell.vs.begin(), cell.vs.end());
				cell.vs.erase(unique(cell.vs.begin(), cell.vs.end()), cell.vs.end());

				for (int j = 0; j < nf; ++j) {
					cell.fs_flag.push_back(1);
				}

				if (last_isolated) {
					cell.v_in_Kernel.push_back(M.vertices.point(nv - 1)[0]);
					cell.v_in_Kernel.push_back(M.vertices.point(nv - 1)[1]);
					cell.v_in_Kernel.push_back(M.vertices.point(nv - 1)[2]);

				} else {
					// Compute a point in the kernel (assumes the barycenter is ok)
					Eigen::RowVector3d p(0, 0, 0);
					for (int v : cell.vs) {
						p += mesh_.points.col(v).transpose();
					}
					p /= cell.vs.size();
					cell.v_in_Kernel.push_back(p[0]);
					cell.v_in_Kernel.push_back(p[1]);
					cell.v_in_Kernel.push_back(p[2]);
				}

			}

			for (int i = 0; i < 1; ++i) {
				mesh_.elements[i].hex = false;
			}//FIME me here!

			mesh_.type = M.cells.are_simplices() ? MeshType::Tet : MeshType::Hyb;
		}
		else {

			// Set faces
			mesh_.faces.clear();
			// for (int f = 0; f < (int) M.facets.nb(); ++f) {
			// 	Face &face = mesh_.faces[f];
			// 	face.id = -1;
			// 	face.vs.clear();
			// 	// face.id = f;

			// 	// face.vs.resize(M.facets.nb_vertices(f));
			// 	// for (int lv = 0; lv < (int) M.facets.nb_vertices(f); ++lv) {
			// 	// 	face.vs[lv] = M.facets.vertex(f, lv);
			// 	// }
			// }

			auto opposite_cell_facet = [&M] (int c, int cf) {
				GEO::index_t c2 = M.cell_facets.adjacent_cell(cf);
				if (c2 == GEO::NO_FACET) { return -1; }
				for (int lf = 0; lf < (int) M.cells.nb_facets(c2); ++lf) {
					// std::cout << M.cells.adjacent(c, lf) << std::endl;
					// std::cout << c << ' ' << M.cells.facet(c2, lf) << std::endl;
					if (c == (int) M.cells.adjacent(c2, lf)) {
						return (int) M.cells.facet(c2, lf);
					}
				}
				assert(false);
				return -1;
			};

			std::vector<int> cell_facet_to_facet(M.cell_facets.nb(), -1);

			// Creates 1 hex or polyhedral element for each cell of the input mesh
			int facet_counter = 0;
			mesh_.elements.resize(M.cells.nb());
			for (int c = 0; c < (int) M.cells.nb(); ++c) {
				Element &cell = mesh_.elements[c];
				cell.id = c;
				cell.hex = (M.cells.type(c) == GEO::MESH_HEX);

				int nf = M.cells.nb_facets(c);
				cell.fs.resize(nf);

				for (int lf = 0; lf < nf; ++lf) {
					int cf = M.cells.facet(c, lf);
					int cf2 = opposite_cell_facet(c, cf);
					// std::cout << "cf2: " << cf2 << std::endl;
					// std::cout << "face_counter: " << facet_counter << std::endl;
					if (cf2 < 0 || cell_facet_to_facet[cf2] < 0) {
						mesh_.faces.emplace_back();
						Face &face = mesh_.faces[facet_counter];
						// std::cout << "fid: " << face.id << std::endl;
						assert(face.vs.empty());
						face.vs.resize(M.cells.facet_nb_vertices(c, lf));
						for (int lv = 0; lv < (int) M.cells.facet_nb_vertices(c, lf); ++lv) {
							face.vs[lv] = M.cells.facet_vertex(c, lf, lv);
						}
						cell.fs_flag.push_back(0);
						cell.fs[lf] = face.id = facet_counter;
						cell_facet_to_facet[cf] = facet_counter;
						++facet_counter;
					} else {
						cell.fs[lf] = cell_facet_to_facet[cf2];
						cell.fs_flag.push_back(1);
					}
				}

				for (auto fid : cell.fs) {
					cell.vs.insert(cell.vs.end(), mesh_.faces[fid].vs.begin(), mesh_.faces[fid].vs.end());
				}
				sort(cell.vs.begin(), cell.vs.end());
				cell.vs.erase(unique(cell.vs.begin(), cell.vs.end()), cell.vs.end());

				// Compute a point in the kernel (assumes the barycenter is ok)
				Eigen::RowVector3d p(0, 0, 0);
				for (int v : cell.vs) {
					p += mesh_.points.col(v).transpose();
				}
				p /= cell.vs.size();
				mesh_.elements[c].v_in_Kernel.push_back(p[0]);
				mesh_.elements[c].v_in_Kernel.push_back(p[1]);
				mesh_.elements[c].v_in_Kernel.push_back(p[2]);
			}
			mesh_.type = M.cells.are_simplices() ? MeshType::Tet : MeshType::Hyb;
		}

		Navigation3D::prepare_mesh(mesh_);
		// if (is_simplicial()) {
		// 	MeshProcessing3D::orient_volume_mesh(mesh_);
		// }
		compute_elements_tag();
		return true;
	}

	bool Mesh3D::save(const std::string &path) const{

		if (!StringUtils::endswidth(path, ".HYBRID")) {
			GEO::Mesh M;
			to_geogram_mesh(*this, M);
			GEO::mesh_save(M, path);
			return true;
		}

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

	bool Mesh3D::build_from_matrices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
	{
		assert(F.cols() == 4 || F.cols() == 8);
		edge_nodes_.clear();
		face_nodes_.clear();
		cell_nodes_.clear();

		GEO::Mesh M;
		M.vertices.create_vertices((int) V.rows());
		for (int i = 0; i < (int) M.vertices.nb(); ++i) {
			GEO::vec3 &p = M.vertices.point(i);
			p[0] = V(i, 0);
			p[1] = V(i, 1);
			p[2] = V(i, 2);
		}

		if (F.cols() == 4)
			M.cells.create_tets((int)F.rows());
		else if (F.cols() == 8)
			M.cells.create_hexes((int)F.rows());
		else
		{
			throw std::runtime_error("Mesh format not supported");
		}

		for (int c = 0; c < (int) M.cells.nb(); ++c) {
			for (int lv = 0; lv < F.cols(); ++lv) {
				M.cells.set_vertex(c, lv, F(c, lv));
			}
		}
		M.cells.connect();

		return load(M);
	}

	void Mesh3D::attach_higher_order_nodes(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &nodes)
	{
		edge_nodes_.clear();
		face_nodes_.clear();
		cell_nodes_.clear();


		edge_nodes_.resize(n_edges());
		face_nodes_.resize(n_faces());
		cell_nodes_.resize(n_cells());

		orders_.resize(n_cells(), 1);

		const auto attach_p2 = [&] (const Navigation3D::Index &index, const std::vector<int> &nodes_ids) {
			auto &n = edge_nodes_[index.edge];

			if(n.nodes.size() > 0)
				return;

			n.v1 = index.vertex;
			n.v2 = switch_vertex(index).vertex;


			const int n_v1 = index.vertex;
			const int n_v2 = switch_vertex(index).vertex;

			int node_index = 0;

			     if((n_v1 == nodes_ids[0] && n_v2 == nodes_ids[1]) || (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[1]))
				node_index = 4;
			else if((n_v1 == nodes_ids[1] && n_v2 == nodes_ids[2]) || (n_v2 == nodes_ids[1] && n_v1 == nodes_ids[2]))
				node_index = 5;
			else if((n_v1 == nodes_ids[2] && n_v2 == nodes_ids[3]) || (n_v2 == nodes_ids[2] && n_v1 == nodes_ids[3]))
				node_index = 8;

			else if((n_v1 == nodes_ids[0] && n_v2 == nodes_ids[3]) || (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[3]))
				node_index = 7;
			else if((n_v1 == nodes_ids[0] && n_v2 == nodes_ids[2]) || (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[2]))
				node_index = 6;
			else
				node_index = 9;

			n.nodes.resize(1, 3);
			n.nodes << V(nodes_ids[node_index], 0), V(nodes_ids[node_index], 1), V(nodes_ids[node_index], 2);
		};

		const auto attach_p3 = [&] (const Navigation3D::Index &index, const std::vector<int> &nodes_ids) {
			auto &n = edge_nodes_[index.edge];

			if(n.nodes.size() > 0)
				return;

			n.v1 = index.vertex;
			n.v2 = switch_vertex(index).vertex;


			const int n_v1 = index.vertex;
			const int n_v2 = switch_vertex(index).vertex;

			int node_index1 = 0;
			int node_index2 = 0;
			if(n_v1 == nodes_ids[0] && n_v2 == nodes_ids[1]) {
				node_index1 = 4;
				node_index2 = 5;
			}
			else if (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[1]) {
				node_index1 = 5;
				node_index2 = 4;
			}
			else if(n_v1 == nodes_ids[1] && n_v2 == nodes_ids[2]) {
				node_index1 = 6;
				node_index2 = 7;
			}
			else if (n_v2 == nodes_ids[1] && n_v1 == nodes_ids[2]) {
				node_index1 = 7;
				node_index2 = 6;
			}
			else if(n_v1 == nodes_ids[2] && n_v2 == nodes_ids[3]) {
				node_index1 = 13;
				node_index2 = 12;
			}
			else if (n_v2 == nodes_ids[2] && n_v1 == nodes_ids[3]) {
				node_index1 = 12;
				node_index2 = 13;
			}

			else if(n_v1 == nodes_ids[0] && n_v2 == nodes_ids[3]) {
				node_index1 = 11;
				node_index2 = 10;
			}
			else if (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[3]) {
				node_index1 = 10;
				node_index2 = 11;
			}
			else if(n_v1 == nodes_ids[0] && n_v2 == nodes_ids[2]) {
				node_index1 = 9;
				node_index2 = 8;
			}
			else if (n_v2 == nodes_ids[0] && n_v1 == nodes_ids[2]) {
				node_index1 = 8;
				node_index2 = 9;
			}

			else if (n_v2 == nodes_ids[1] && n_v1 == nodes_ids[3]) {
				node_index1 = 14;
				node_index2 = 15;
			}
			else{
				node_index1 = 15;
				node_index2 = 14;
			}

			n.nodes.resize(2, 3);
			n.nodes.row(0) << V(nodes_ids[node_index1], 0), V(nodes_ids[node_index1], 1), V(nodes_ids[node_index1], 2);
			n.nodes.row(1) << V(nodes_ids[node_index2], 0), V(nodes_ids[node_index2], 1), V(nodes_ids[node_index2], 2);
		};

		const auto attach_p3_face = [&] (const Navigation3D::Index &index, const std::vector<int> &nodes_ids, int id) {
			auto &n = face_nodes_[index.face];
			if(n.nodes.size() <= 0)
			{
				n.v1 = face_vertex(index.face, 0);
				n.v2 = face_vertex(index.face, 1);
				n.v3 = face_vertex(index.face, 2);
				n.nodes.resize(1, 3);
				n.nodes << V(nodes_ids[id], 0), V(nodes_ids[id], 1), V(nodes_ids[id], 2);
			}
		};

		assert(nodes.size() == n_cells());

		for(int c = 0; c < n_cells(); ++c)
		{
			auto index = get_index_from_element(c);

			const auto &nodes_ids = nodes[c];

			if(nodes_ids.size() == 4){
				orders_(c) = 1;
				continue;
			}
			//P2
			else if(nodes_ids.size() == 10)
			{
				orders_(c) = 2;

				for(int le = 0; le < 3; ++le)
				{
					attach_p2(index, nodes_ids);
					index = next_around_face(index);
				}

				index = switch_vertex(switch_edge(switch_face(index)));
				attach_p2(index, nodes_ids);

				index = switch_edge(index);
				attach_p2(index, nodes_ids);

				index = switch_edge(switch_face(index));
				attach_p2(index, nodes_ids);
			}
			//P3
			else if(nodes_ids.size() == 20)
			{
				orders_(c) = 3;

				for(int le = 0; le < 3; ++le)
				{
					attach_p3(index, nodes_ids);
					index = next_around_face(index);
				}

				{
					index = switch_vertex(switch_edge(switch_face(index)));
					attach_p3(index, nodes_ids);

					index = switch_edge(index);
					attach_p3(index, nodes_ids);

					index = switch_edge(switch_face(index));
					attach_p3(index, nodes_ids);
				}

				{
					index = get_index_from_element(c);

					attach_p3_face(index, nodes_ids, 19);
					attach_p3_face(switch_face(index), nodes_ids, 17);
					attach_p3_face(switch_face(next_around_face(index)), nodes_ids, 18);
					attach_p3_face(switch_face(next_around_face(next_around_face(index))), nodes_ids, 16);
				}
			}
			//P4
			else if(nodes_ids.size() == 15)
			{
				orders_(c) = 4;
				assert(false);
				// unsupported P4 for geometry, need meshes for testing
			}
			//unsupported
			else
			{
				assert(false);
			}
		}
	}

	RowVectorNd Mesh3D::edge_node(const Navigation3D::Index &index, const int n_new_nodes, const int i) const
	{
		if(orders_.size() <= 0 || orders_(index.element) == 1 || edge_nodes_.empty() || edge_nodes_[index.edge].nodes.rows() != n_new_nodes)
		{
			const auto v1 = point(index.vertex);
			const auto v2 = point(switch_vertex(index).vertex);

			const double t = i/(n_new_nodes + 1.0);

			return (1 - t) * v1 + t * v2;
		}

		const auto &n = edge_nodes_[index.edge];
		if(n.v1 == index.vertex)
			return n.nodes.row(i-1);
		else
			return n.nodes.row(n.nodes.rows() - i);
	}

	RowVectorNd Mesh3D::face_node(const Navigation3D::Index &index, const int n_new_nodes, const int i, const int j) const
	{
		if(is_simplex(index.element))
		{
			if(orders_.size() <= 0 || orders_(index.element) == 1 || orders_(index.element) == 2 || face_nodes_.empty() || face_nodes_[index.face].nodes.rows() != n_new_nodes)
			{
				const auto v1 = point(index.vertex);
				const auto v2 = point(switch_vertex(index).vertex);
				const auto v3 = point(switch_vertex(switch_edge(index)).vertex);

				const double b2 = i/(n_new_nodes + 2.0);
				const double b3 = j/(n_new_nodes + 2.0);
				const double b1 = 1 - b3 - b2;
				assert(b3 < 1);
				assert(b3 > 0);

				return b1 * v1 + b2 * v2 + b3 * v3;
			}

			assert(orders_(index.element) == 3);
			//unsupported P4 for geometry
			const auto &n = face_nodes_[index.face];
			return n.nodes.row(0);
		}
		else if(is_cube(index.element))
		{
			//supports only blilinear quads
			assert(orders_.size() <= 0 || orders_(index.element) == 1);

			const auto v1 = point(index.vertex);
			const auto v2 = point(switch_vertex(index).vertex);
			const auto v3 = point(switch_vertex(switch_edge(switch_vertex(index))).vertex);
			const auto v4 = point(switch_vertex(switch_edge(index)).vertex);

			const double b1 = i/(n_new_nodes + 1.0);
			const double b2 = j/(n_new_nodes + 1.0);

			return v1*(1-b1)*(1-b2) + v2*b1*(1-b2) + v3*b1*b2 + v4*(1-b1)*b2;
		}

		assert(false);
		return RowVectorNd(3,1);
	}

	RowVectorNd Mesh3D::cell_node(const Navigation3D::Index &index, const int n_new_nodes, const int i, const int j, const int k) const
	{
		if(n_new_nodes == 1)
			return cell_barycenter(index.element);

		if(is_simplex(index.element))
		{
			assert(n_new_nodes == 1);
			return cell_barycenter(index.element);
		}
		else if(is_cube(index.element))
		{
			//supports only blilinear quads
			assert(orders_.size() <= 0 || orders_(index.element) == 1);

			const auto v1 = point(index.vertex);
			const auto v2 = point(switch_vertex(index).vertex);
			const auto v3 = point(switch_vertex(switch_edge(switch_vertex(index))).vertex);
			const auto v4 = point(switch_vertex(switch_edge(index)).vertex);

			const Navigation3D::Index index1 = switch_face(switch_edge(switch_vertex(switch_edge(switch_face(index)))));
			const auto v5 = point(index1.vertex);
			const auto v6 = point(switch_vertex(index1).vertex);
			const auto v7 = point(switch_vertex(switch_edge(switch_vertex(index1))).vertex);
			const auto v8 = point(switch_vertex(switch_edge(index1)).vertex);

			const double b1 = i/(n_new_nodes + 1.0);
			const double b2 = j/(n_new_nodes + 1.0);

			const double b3 = k/(n_new_nodes + 1.0);

			RowVectorNd blin1 = v1*(1-b1)*(1-b2) + v2*b1*(1-b2) + v3*b1*b2 + v4*(1-b1)*b2;
			RowVectorNd blin2 = v5*(1-b1)*(1-b2) + v6*b1*(1-b2) + v7*b1*b2 + v8*(1-b1)*b2;

			return (1-b3)*blin1 + b3*blin2;
		}

		assert(false);
		return RowVectorNd(3,1);
	}

	void Mesh3D::bounding_box(RowVectorNd &min, RowVectorNd &max) const
	{
		const auto &V = mesh_.points;
		min = V.rowwise().minCoeff().transpose();
		max = V.rowwise().maxCoeff().transpose();
	}

	void Mesh3D::normalize() {
		auto &V = mesh_.points;
		Eigen::RowVector3d minV = V.rowwise().minCoeff().transpose();
		Eigen::RowVector3d maxV = V.rowwise().maxCoeff().transpose();
		const auto shift =  V.rowwise().minCoeff().eval();
		const double scaling = 1.0 / (V.rowwise().maxCoeff() - V.rowwise().minCoeff()).maxCoeff();
		V = (V.colwise() - shift) * scaling;

		for(int i = 0; i < n_cells(); ++i)
		{
			for(int d = 0; d < 3; ++d){
				auto val = mesh_.elements[i].v_in_Kernel[d];
				mesh_.elements[i].v_in_Kernel[d] = (val - shift(d)) * scaling;
			}
		}

		for(auto &n : edge_nodes_){
			if(n.nodes.size() > 0)
				n.nodes = (n.nodes.rowwise() - shift.transpose()) * scaling;
		}
		for(auto &n : face_nodes_){
			if(n.nodes.size() > 0)
				n.nodes = (n.nodes.rowwise() - shift.transpose()) * scaling;
		}
		for(auto &n : cell_nodes_){
			if(n.nodes.size() > 0)
				n.nodes = (n.nodes.rowwise() - shift.transpose()) * scaling;
		}

		logger().debug("-- bbox before normalization:");
		logger().debug("   min   : {} {} {}", minV(0), minV(1), minV(2));
		logger().debug("   max   : {} {} {}", maxV(0), maxV(1), maxV(2));
		logger().debug("   extent: {} {} {}", maxV(0) - minV(0), maxV(1) - minV(1), maxV(2) - minV(2));
		minV = V.rowwise().minCoeff().transpose();
		maxV = V.rowwise().maxCoeff().transpose();
		logger().debug("-- bbox after normalization:");
		logger().debug("   min   : {} {} {}", minV(0), minV(1), minV(2));
		logger().debug("   max   : {} {} {}", maxV(0), maxV(1), maxV(2));
		logger().debug("   extent: {} {} {}", maxV(0) - minV(0), maxV(1) - minV(1), maxV(2) - minV(2));

		// V.row(1) /= 100.;

		// for(int i = 0; i < n_cells(); ++i)
		// {
		// 	mesh_.elements[i].v_in_Kernel[1] /= 100.;
		// }

		Eigen::MatrixXd p0, p1, p;
		get_edges(p0, p1);
		p = p0 - p1;
		logger().debug("-- edge length after normalization:");
		logger().debug("   min: ", p.rowwise().norm().minCoeff());
		logger().debug("   max: ", p.rowwise().norm().maxCoeff());
		logger().debug("   avg: ", p.rowwise().norm().mean());
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

	bool Mesh3D::is_boundary_element(const int element_global_id) const
	{
		const auto &fs = mesh_.elements[element_global_id].fs;

		for(auto f_id : fs)
		{
			if(is_boundary_face(f_id))
				return true;
		}

		const auto &vs = mesh_.elements[element_global_id].vs;

		for(auto v_id : vs)
		{
			if(is_boundary_vertex(v_id))
				return true;
		}

		return false;
	}

	void Mesh3D::compute_boundary_ids(const std::function<int(const RowVectorNd&)> &marker)
	{
		boundary_ids_.resize(n_faces());
		std::fill(boundary_ids_.begin(), boundary_ids_.end(), -1);

		for(int f = 0; f < n_faces(); ++f)
		{
			if(!is_boundary_face(f))
				continue;

			const auto p = face_barycenter(f);
			boundary_ids_[f]=marker(p);
		}
	}

	void Mesh3D::compute_boundary_ids(const std::function<int(const RowVectorNd&, bool)> &marker)
	{
		boundary_ids_.resize(n_faces());

		for(int f = 0; f < n_faces(); ++f)
		{
			const bool is_boundary = is_boundary_face(f);
			const auto p = face_barycenter(f);
			boundary_ids_[f]=marker(p, is_boundary);
		}
	}

	void Mesh3D::compute_boundary_ids(const std::function<int(const std::vector<int>&, bool)> &marker)
	{
		boundary_ids_.resize(n_faces());

		for(int f = 0; f < n_faces(); ++f)
		{
			const bool is_boundary = is_boundary_face(f);
			std::vector<int> vs(n_face_vertices(f));
			for(int vid = 0; vid < vs.size(); ++vid)
				vs[vid] = face_vertex(f,vid);

			std::sort(vs.begin(), vs.end());
			boundary_ids_[f]=marker(vs, is_boundary);
		}
	}

	void Mesh3D::compute_boundary_ids(const double eps)
	{
		boundary_ids_.resize(n_faces());
		std::fill(boundary_ids_.begin(), boundary_ids_.end(), -1);

		const auto &V = mesh_.points;
		Eigen::RowVector3d minV = V.rowwise().minCoeff().transpose();
		Eigen::RowVector3d maxV = V.rowwise().maxCoeff().transpose();

		for(int f = 0; f < n_faces(); ++f)
		{
			if(!is_boundary_face(f))
				continue;

			const auto p = face_barycenter(f);

			if(fabs(p(0)-minV(0))<eps)
				boundary_ids_[f]=1;
			else if(fabs(p(1)-minV(1))<eps)
				boundary_ids_[f]=2;
			else if(fabs(p(2)-minV(2))<eps)
				boundary_ids_[f]=5;

			else if(fabs(p(0)-maxV(0))<eps)
				boundary_ids_[f]=3;
			else if(fabs(p(1)-maxV(1))<eps)
				boundary_ids_[f]=4;
			else if(fabs(p(2)-maxV(2))<eps)
				boundary_ids_[f]=6;
			else
				boundary_ids_[f]=7;
		}


	}

	RowVectorNd Mesh3D::point(const int global_index) const {
		RowVectorNd pt = mesh_.points.col(global_index).transpose();
		return pt;
	}

	RowVectorNd Mesh3D::kernel(const int c) const {
		RowVectorNd pt(3);
		pt << mesh_.elements[c].v_in_Kernel[0], mesh_.elements[c].v_in_Kernel[1], mesh_.elements[c].v_in_Kernel[2];
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

	void Mesh3D::get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1, const std::vector<bool> &valid_elements) const
	{
		int count = 0;
		for(size_t i = 0; i < valid_elements.size(); ++i)
		{
			if(valid_elements[i]){
				count += mesh_.elements[i].es.size();
			}
		}

		p0.resize(count, 3);
		p1.resize(count, 3);

		count = 0;

		for(size_t i = 0; i < valid_elements.size(); ++i)
		{
			if(!valid_elements[i])
				continue;

			for(size_t ei = 0; ei < mesh_.elements[i].es.size(); ++ei)
			{
				const int e = mesh_.elements[i].es[ei];
				p0.row(count) = point(mesh_.edges[e].vs[0]);
				p1.row(count) = point(mesh_.edges[e].vs[1]);

				++count;
			}
		}
	}


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
				if (attaching_non_hex) {
					ele_tag[ele.id] = ElementType::InterfaceCube;
					continue;
				}

				if (on_boundary) {
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
						if (n_irregular_e == 1) {
							ele_tag[ele.id] = ElementType::SimpleSingularBoundaryCube;
						}
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


		//TODO correct?
		for (auto &ele : mesh_.elements) {
			if(ele.vs.size() == 4)
				ele_tag[ele.id] = ElementType::Simplex;
		}
	}

	double Mesh3D::quad_area(const int gid) const
	{
		const int n_vertices = n_face_vertices(gid);
		assert(n_vertices == 4);

		const auto &vertices = mesh_.faces[gid].vs;

		const auto v1 = point(vertices[0]);
		const auto v2 = point(vertices[1]);
		const auto v3 = point(vertices[2]);
		const auto v4 = point(vertices[3]);

		const Vector3d e0 = (v2 - v1).transpose();
		const Vector3d e1 = (v3 - v1).transpose();

		const Vector3d e2 = (v2 - v4).transpose();
		const Vector3d e3 = (v3 - v4).transpose();

		return e0.cross(e1).norm()/2 + e2.cross(e3).norm()/2;
	}

	double Mesh3D::tri_area(const int gid) const
	{
		const int n_vertices = n_face_vertices(gid);
		assert(n_vertices == 3);

		const auto &vertices = mesh_.faces[gid].vs;

		const auto v1 = point(vertices[0]);
		const auto v2 = point(vertices[1]);
		const auto v3 = point(vertices[2]);

		const Vector3d e0 = (v2 - v1).transpose();
		const Vector3d e1 = (v3 - v1).transpose();

		return e0.cross(e1).norm()/2;
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
		to_face[0]= [&](Navigation3D::Index idx) { return switch_face(switch_edge(switch_vertex(switch_edge(switch_face(idx))))); };
        //bottom
		to_face[1]= [&](Navigation3D::Index idx) { return idx; };

        //left
		to_face[2]= [&](Navigation3D::Index idx) { return switch_face(switch_edge(switch_vertex(idx))); };
        //right
		to_face[3]= [&](Navigation3D::Index idx) { return switch_face(switch_edge(idx)); };

        //back
		to_face[4]= [&](Navigation3D::Index idx) { return switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(idx))))); };
        //front
		to_face[5]= [&](Navigation3D::Index idx) { return switch_face(idx); };
	}

	void Mesh3D::to_vertex_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 8> &to_vertex) const
	{
		to_vertex[0]= [&](Navigation3D::Index idx) { return idx; };
		to_vertex[1]= [&](Navigation3D::Index idx) { return switch_vertex(idx); };
		to_vertex[2]= [&](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_vertex(idx))); };
		to_vertex[3]= [&](Navigation3D::Index idx) { return switch_vertex(switch_edge(idx)); };

		to_vertex[4]= [&](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_face(idx))); };
		to_vertex[5]= [&](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_vertex(switch_edge(switch_face(idx))))); };
		to_vertex[6]= [&](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_face(switch_vertex(switch_edge(switch_vertex(idx)))))); };
		to_vertex[7]= [&](Navigation3D::Index idx) { return switch_vertex(switch_edge(switch_face(switch_vertex(switch_edge(idx))))); };
	}

	void Mesh3D::to_edge_functions(std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 12> &to_edge) const
	{
		to_edge[0]= [&](Navigation3D::Index idx) { return idx; };
		to_edge[1]= [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(idx)); };
		to_edge[2]= [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))); };
		to_edge[3]= [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))); };

		to_edge[4]= [&](Navigation3D::Index idx) { return switch_edge(switch_face(idx)); };
		to_edge[5]= [&](Navigation3D::Index idx) { return switch_edge(switch_face(switch_edge(switch_vertex(idx)))); };
		to_edge[6]= [&](Navigation3D::Index idx) { return switch_edge(switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))); };
		to_edge[7]= [&](Navigation3D::Index idx) { return switch_edge(switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))))); };

		to_edge[8]= [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_face(idx)))); };
		to_edge[9]= [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_face(switch_edge(switch_vertex(idx)))))); };
		to_edge[10]= [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))))); };
		to_edge[11]= [&](Navigation3D::Index idx) { return switch_edge(switch_vertex(switch_edge(switch_face(switch_edge(switch_vertex(switch_edge(switch_vertex(switch_edge(switch_vertex(idx)))))))))); };
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


		// std::array<GEO::vec3, 4> vertices;

		// for(int lv = 0; lv < 4; ++lv)
		// {
		// 	auto pt = point(v[lv]);
		// 	for(int d = 0; d < 3; ++d)
		// 	{
		// 		vertices[lv][d] = pt(d);
		// 	}
		// }

		// const double vol = GEO::Geom::tetra_signed_volume(vertices[0], vertices[1], vertices[2], vertices[3]);
		// if(vol < 0)
		// {
		// 	std::cout << "negative vol" << std::endl;
		// //	idx = switch_vertex(get_index_from_element(element_index));
		// //	for (int lv = 0; lv < 3; ++lv) {
		// //		v[lv] = idx.vertex;
		// //		idx = next_around_face(idx);
		// //	}
		// //// assert(idx == get_index_from_element(element_index));
		// //	idx = switch_vertex(switch_edge(switch_face(idx)));
		// //	v[3] = idx.vertex;
		// }


		return v;
	}











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

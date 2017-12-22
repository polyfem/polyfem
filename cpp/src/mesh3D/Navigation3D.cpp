#include "Navigation3D.hpp"
#include <algorithm>
#include <iterator>
#include <set>
#include <cassert>

using namespace poly_fem::Navigation3D;
using namespace poly_fem;
using namespace std;


namespace
{
	bool predicate(const uint32_t a, const uint32_t b){ return a == b; }


	void build_connectivity(Mesh3DStorage &hmi) {
		hmi.edges.clear();
		if (hmi.type == MeshType::Hyb) {
			vector<bool> bf_flag(hmi.faces.size(), false);
			for (auto h : hmi.elements) for (auto f : h.fs)bf_flag[f] = !bf_flag[f];
				for (auto &f : hmi.faces) f.boundary = bf_flag[f.id];

					std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> temp;
				for (uint32_t i = 0; i < hmi.faces.size(); ++i) {
					int fl = hmi.faces[i].vs.size();
					for (uint32_t j = 0; j < hmi.faces[i].vs.size(); ++j) {
						uint32_t v0 = hmi.faces[i].vs[j], v1 = hmi.faces[i].vs[(j + 1) % fl];
						if (v0 > v1) std::swap(v0, v1);
						temp.push_back(std::make_tuple(v0, v1, i, j));
					}
					hmi.faces[i].es.resize(fl);
				}
				std::sort(temp.begin(), temp.end());
				hmi.edges.reserve(temp.size() / 2);
				uint32_t E_num = 0;
				Edge e; e.boundary = false; e.vs.resize(2);
				for (uint32_t i = 0; i < temp.size(); ++i) {
					if (i == 0 || (i != 0 && (std::get<0>(temp[i]) != std::get<0>(temp[i - 1]) ||
						std::get<1>(temp[i]) != std::get<1>(temp[i - 1])))) {
						e.id = E_num; E_num++;
					e.vs[0] = std::get<0>(temp[i]);
					e.vs[1] = std::get<1>(temp[i]);
					hmi.edges.push_back(e);
				}
				hmi.faces[std::get<2>(temp[i])].es[std::get<3>(temp[i])] = E_num - 1;
			}
		//boundary
			for (auto &v : hmi.vertices) v.boundary = false;
				for (uint32_t i = 0; i < hmi.faces.size(); ++i)
					if (hmi.faces[i].boundary) for (uint32_t j = 0; j < hmi.faces[i].vs.size(); ++j) {
						uint32_t eid = hmi.faces[i].es[j];
						hmi.edges[eid].boundary = true;
						hmi.vertices[hmi.faces[i].vs[j]].boundary = true;
					}
				}
	//f_nhs;
				for (uint32_t i = 0; i < hmi.elements.size(); i++) {
					for (uint32_t j = 0; j < hmi.elements[i].fs.size(); j++) hmi.faces[hmi.elements[i].fs[j]].neighbor_hs.push_back(i);
				}
	//e_nfs, v_nfs
			for (uint32_t i = 0; i < hmi.faces.size(); i++) {
				for (uint32_t j = 0; j < hmi.faces[i].es.size(); j++) hmi.edges[hmi.faces[i].es[j]].neighbor_fs.push_back(i);
					for (uint32_t j = 0; j < hmi.faces[i].vs.size(); j++) hmi.vertices[hmi.faces[i].vs[j]].neighbor_fs.push_back(i);
				}
	//v_nes, v_nvs
			for (uint32_t i = 0; i < hmi.edges.size(); i++) {
				uint32_t v0 = hmi.edges[i].vs[0], v1 = hmi.edges[i].vs[1];
				hmi.vertices[v0].neighbor_es.push_back(i);
				hmi.vertices[v1].neighbor_es.push_back(i);
				hmi.vertices[v0].neighbor_vs.push_back(v1);
				hmi.vertices[v1].neighbor_vs.push_back(v0);
			}
	//e_nhs
			for (uint32_t i = 0; i < hmi.edges.size(); i++) {
				std::vector<uint32_t> nhs;
				for (uint32_t j = 0; j < hmi.edges[i].neighbor_fs.size(); j++) {
					uint32_t nfid = hmi.edges[i].neighbor_fs[j];
					nhs.insert(nhs.end(), hmi.faces[nfid].neighbor_hs.begin(), hmi.faces[nfid].neighbor_hs.end());
				}
				std::sort(nhs.begin(), nhs.end()); nhs.erase(std::unique(nhs.begin(), nhs.end()), nhs.end());
				hmi.edges[i].neighbor_hs = nhs;
			}
//v_nhs; ordering fs for hex
			for (uint32_t i = 0; i < hmi.elements.size(); i++) {
				vector<uint32_t> vs;
				for (auto fid : hmi.elements[i].fs)vs.insert(vs.end(), hmi.faces[fid].vs.begin(), hmi.faces[fid].vs.end());
					sort(vs.begin(), vs.end()); vs.erase(unique(vs.begin(), vs.end()), vs.end());


				if (hmi.elements[i].hex) {
					int top_fid = hmi.elements[i].fs[0];
					hmi.elements[i].vs = hmi.faces[top_fid].vs;

					std::set<uint32_t> s_model(vs.begin(), vs.end());
					std::set<uint32_t> s_pattern(hmi.faces[top_fid].vs.begin(), hmi.faces[top_fid].vs.end());
					vector<uint32_t> vs_left;
					std::set_difference(s_model.begin(), s_model.end(), s_pattern.begin(), s_pattern.end(), std::back_inserter(vs_left));

					for (auto vid : hmi.faces[top_fid].vs)for (auto nvid : hmi.vertices[vid].neighbor_vs)
						if (find(vs_left.begin(), vs_left.end(), nvid) != vs_left.end()) {
							hmi.elements[i].vs.push_back(nvid); break;
						}

						function<int(vector<uint32_t> &)> WHICH_F = [&](vector<uint32_t> &vs0)->int {
							int which_f = -1;
							sort(vs0.begin(), vs0.end());
							bool found_f = false;
							for (auto fid : hmi.elements[i].fs) {
								vector<uint32_t> vs1 = hmi.faces[fid].vs;
								sort(vs1.begin(), vs1.end());
								if (vs0.size() == vs1.size() && std::equal(vs0.begin(), vs0.end(), vs1.begin())) {
									which_f = fid; break;
								}
							}
							return which_f;
						};

						vector<uint32_t> fs;
						fs.push_back(top_fid);
						vector<uint32_t> vs_temp;

						vs_temp.insert(vs_temp.end(), hmi.elements[i].vs.begin() + 4, hmi.elements[i].vs.end());
						int bottom_fid = WHICH_F(vs_temp);
						fs.push_back(bottom_fid);

						vs_temp.clear();
						vs_temp.push_back(hmi.elements[i].vs[0]);
						vs_temp.push_back(hmi.elements[i].vs[1]);
						vs_temp.push_back(hmi.elements[i].vs[4]);
						vs_temp.push_back(hmi.elements[i].vs[5]);
						int front_fid = WHICH_F(vs_temp);
						fs.push_back(front_fid);

						vs_temp.clear();
						vs_temp.push_back(hmi.elements[i].vs[2]);
						vs_temp.push_back(hmi.elements[i].vs[3]);
						vs_temp.push_back(hmi.elements[i].vs[6]);
						vs_temp.push_back(hmi.elements[i].vs[7]);
						int back_fid = WHICH_F(vs_temp);
						fs.push_back(back_fid);

						vs_temp.clear();
						vs_temp.push_back(hmi.elements[i].vs[1]);
						vs_temp.push_back(hmi.elements[i].vs[2]);
						vs_temp.push_back(hmi.elements[i].vs[5]);
						vs_temp.push_back(hmi.elements[i].vs[6]);
						int left_fid = WHICH_F(vs_temp);
						fs.push_back(left_fid);

						vs_temp.clear();
						vs_temp.push_back(hmi.elements[i].vs[3]);
						vs_temp.push_back(hmi.elements[i].vs[0]);
						vs_temp.push_back(hmi.elements[i].vs[7]);
						vs_temp.push_back(hmi.elements[i].vs[4]);
						int right_fid = WHICH_F(vs_temp);
						fs.push_back(right_fid);

						hmi.elements[i].fs = fs;
					}
					else hmi.elements[i].vs = vs;

					for (uint32_t j = 0; j < hmi.elements[i].vs.size(); j++) hmi.vertices[hmi.elements[i].vs[j]].neighbor_hs.push_back(i);
				}
		}
	}

	void poly_fem::Navigation3D::prepare_mesh(Mesh3DStorage &M) {
		M.type = MeshType::Hyb;
		build_connectivity(M);
	}


	Index poly_fem::Navigation3D::get_index_from_element_face(const Mesh3DStorage &M, int hi, int lf, int lv)
	{
		Index idx;

		if (hi >= M.elements.size()) hi = hi % M.elements.size();
		idx.element = hi;

		if (lf >= M.elements[hi].fs.size()) lf = lf % M.elements[hi].fs.size();
		idx.element_patch = lf;
		idx.face = M.elements[hi].fs[idx.element_patch];

		if (lv >= M.faces[idx.face].vs.size()) lv = lv % M.faces[idx.face].vs.size();
		idx.face_corner = lv;
		if (!M.elements[hi].fs_flag[idx.element_patch]) idx.face_corner = M.faces[idx.face].vs.size() - 1 - idx.face_corner;
		idx.vertex = M.faces[idx.face].vs[idx.face_corner];
		idx.edge = M.faces[idx.face].es[idx.face_corner];

		return idx;
	}
// Navigation in a surface mesh
	Index poly_fem::Navigation3D::switch_vertex(const Mesh3DStorage &M, Index idx) {

		if(idx.vertex == M.edges[idx.edge].vs[0])idx.vertex = M.edges[idx.edge].vs[1];
		else idx.vertex = M.edges[idx.edge].vs[0];
		idx.face_corner = std::find(M.faces[idx.face].vs.begin(), M.faces[idx.face].vs.end(), idx.vertex) - M.faces[idx.face].vs.begin();
		if (!M.elements[idx.element].fs_flag[idx.element_patch]) idx.face_corner = M.faces[idx.face].vs.size() - 1 - idx.face_corner;

		return idx;
	}
	Index poly_fem::Navigation3D::switch_edge(const Mesh3DStorage &M, Index idx) {

		vector<uint32_t> ves = M.vertices[idx.vertex].neighbor_es, fes = M.faces[idx.face].es, sharedes;
		sort(fes.begin(), fes.end()); sort(ves.begin(), ves.end());
		set_intersection(fes.begin(), fes.end(), ves.begin(), ves.end(), back_inserter(sharedes));
	assert(sharedes.size() == 2);//true for sure
	if (sharedes[0] == idx.edge) idx.edge = sharedes[1];else idx.edge = sharedes[0];

	return idx;
}
Index poly_fem::Navigation3D::switch_face(const Mesh3DStorage &M, Index idx) {

	vector<uint32_t> efs = M.edges[idx.edge].neighbor_fs, hfs = M.elements[idx.element].fs, sharedfs;
	sort(hfs.begin(), hfs.end()); sort(efs.begin(), efs.end());
	set_intersection(hfs.begin(), hfs.end(), efs.begin(), efs.end(), back_inserter(sharedfs));
	assert(sharedfs.size() == 2);//true for sure
	if (sharedfs[0] == idx.face) idx.face = sharedfs[1]; else idx.face = sharedfs[0];
	idx.face_corner = std::find(M.faces[idx.face].vs.begin(), M.faces[idx.face].vs.end(), idx.vertex) - M.faces[idx.face].vs.begin();
	if (!M.elements[idx.element].fs_flag[idx.element_patch]) idx.face_corner = M.faces[idx.face].vs.size() - 1 - idx.face_corner;

	return idx;
}
Index poly_fem::Navigation3D::switch_element(const Mesh3DStorage &M, Index idx) {

	if (M.faces[idx.face].neighbor_hs.size() == 1) {
		idx.element = -1;
		return idx;
	}
	else {
		if (M.faces[idx.face].neighbor_hs[0] == idx.element)
			idx.element = M.faces[idx.face].neighbor_hs[1];
		else idx.element = M.faces[idx.face].neighbor_hs[0];
	}
	idx.face_corner = std::find(M.faces[idx.face].vs.begin(), M.faces[idx.face].vs.end(), idx.vertex) - M.faces[idx.face].vs.begin();
	if (!M.elements[idx.element].fs_flag[idx.element_patch]) idx.face_corner = M.faces[idx.face].vs.size() - 1 - idx.face_corner;

	return idx;
}
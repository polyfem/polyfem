#include "Navigation3D.hpp"
#include <algorithm>
#include <iterator>
#include <cassert>
using namespace poly_fem::Navigation3D;

void poly_fem::Navigation3D::prepare_mesh(Mesh &M) {
	M.type = Mesh_type::Hyb;
	build_connectivity(M);
}
void poly_fem::Navigation3D::build_connectivity(Mesh &hmi) {
	hmi.Es.clear();
	if (hmi.type == Mesh_type::Hyb) {
		vector<bool> bf_flag(hmi.Fs.size(), false);
		for (auto h : hmi.Hs) for (auto f : h.fs)bf_flag[f] = !bf_flag[f];
		for (auto &f : hmi.Fs) f.boundary = bf_flag[f.id];

		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> temp;
		for (uint32_t i = 0; i < hmi.Fs.size(); ++i) {
			int fl = hmi.Fs[i].vs.size();
			for (uint32_t j = 0; j < hmi.Fs[i].vs.size(); ++j) {
				uint32_t v0 = hmi.Fs[i].vs[j], v1 = hmi.Fs[i].vs[(j + 1) % fl];
				if (v0 > v1) std::swap(v0, v1);
				temp.push_back(std::make_tuple(v0, v1, i, j));
			}
			hmi.Fs[i].es.resize(fl);
		}
		std::sort(temp.begin(), temp.end());
		hmi.Es.reserve(temp.size() / 2);
		uint32_t E_num = 0;
		Hybrid_E e; e.boundary = false; e.vs.resize(2);
		for (uint32_t i = 0; i < temp.size(); ++i) {
			if (i == 0 || (i != 0 && (std::get<0>(temp[i]) != std::get<0>(temp[i - 1]) ||
				std::get<1>(temp[i]) != std::get<1>(temp[i - 1])))) {
				e.id = E_num; E_num++;
				e.vs[0] = std::get<0>(temp[i]);
				e.vs[1] = std::get<1>(temp[i]);
				hmi.Es.push_back(e);
			}
			hmi.Fs[std::get<2>(temp[i])].es[std::get<3>(temp[i])] = E_num - 1;
		}
		//boundary
		for (auto &v : hmi.Vs) v.boundary = false;
		for (uint32_t i = 0; i < hmi.Fs.size(); ++i)
			if (hmi.Fs[i].boundary) for (uint32_t j = 0; j < hmi.Fs[i].vs.size(); ++j) {
				uint32_t eid = hmi.Fs[i].es[j];
				hmi.Es[eid].boundary = true;
				hmi.Vs[hmi.Fs[i].vs[j]].boundary = true;
			}
	}
	//f_nhs;
	for (uint32_t i = 0; i < hmi.Hs.size(); i++) {
		for (uint32_t j = 0; j < hmi.Hs[i].fs.size(); j++) hmi.Fs[hmi.Hs[i].fs[j]].neighbor_hs.push_back(i);
	}
	//e_nfs, v_nfs
	for (uint32_t i = 0; i < hmi.Fs.size(); i++) {
		for (uint32_t j = 0; j < hmi.Fs[i].es.size(); j++) hmi.Es[hmi.Fs[i].es[j]].neighbor_fs.push_back(i);
		for (uint32_t j = 0; j < hmi.Fs[i].vs.size(); j++) hmi.Vs[hmi.Fs[i].vs[j]].neighbor_fs.push_back(i);
	}
	//v_nes, v_nvs
	for (uint32_t i = 0; i < hmi.Es.size(); i++) {
		uint32_t v0 = hmi.Es[i].vs[0], v1 = hmi.Es[i].vs[1];
		hmi.Vs[v0].neighbor_es.push_back(i);
		hmi.Vs[v1].neighbor_es.push_back(i);
		hmi.Vs[v0].neighbor_vs.push_back(v1);
		hmi.Vs[v1].neighbor_vs.push_back(v0);
	}
	//e_nhs
	for (uint32_t i = 0; i < hmi.Es.size(); i++) {
		std::vector<uint32_t> nhs;
		for (uint32_t j = 0; j < hmi.Es[i].neighbor_fs.size(); j++) {
			uint32_t nfid = hmi.Es[i].neighbor_fs[j];
			nhs.insert(nhs.end(), hmi.Fs[nfid].neighbor_hs.begin(), hmi.Fs[nfid].neighbor_hs.end());
		}
		std::sort(nhs.begin(), nhs.end()); nhs.erase(std::unique(nhs.begin(), nhs.end()), nhs.end());
		hmi.Es[i].neighbor_hs = nhs;
	}
}

Index poly_fem::Navigation3D::get_index_from_hedraface(const Mesh &M, int hi, int lf, int lv) {


	Index idx;

	if (hi > M.Hs.size()) hi = hi % M.Hs.size();
	idx.hedra = hi;

	if (lf > M.Hs[hi].fs.size()) lf = lf % M.Hs[hi].fs.size();
	idx.hedra_patch = lf;
	idx.face = M.Hs[hi].fs[idx.hedra_patch];

	if (lv > M.Fs[idx.face].vs.size()) lv = lv % M.Fs[idx.face].vs.size();
	idx.face_corner = lv;
	if (!M.Hs[hi].fs_flag[idx.hedra_patch]) idx.face_corner = M.Fs[idx.face].vs.size() - 1 - idx.face_corner;
	idx.vertex = M.Fs[idx.face].vs[idx.face_corner];
	idx.edge = M.Fs[idx.face].es[idx.face_corner];

	return idx;
}
// Navigation in a surface mesh
Index poly_fem::Navigation3D::switch_vertex(const Mesh &M, Index idx) {

	if(idx.vertex == M.Es[idx.edge].vs[0])idx.vertex = M.Es[idx.edge].vs[1];
	else idx.vertex = M.Es[idx.edge].vs[0];
	idx.face_corner = std::find(M.Fs[idx.face].vs.begin(), M.Fs[idx.face].vs.end(), idx.vertex) - M.Fs[idx.face].vs.begin();
	if (!M.Hs[idx.hedra].fs_flag[idx.hedra_patch]) idx.face_corner = M.Fs[idx.face].vs.size() - 1 - idx.face_corner;

	return idx;
}
Index poly_fem::Navigation3D::switch_edge(const Mesh &M, Index idx) {

	vector<uint32_t> ves = M.Vs[idx.vertex].neighbor_es, fes = M.Fs[idx.face].es, sharedes;
	sort(fes.begin(), fes.end()); sort(ves.begin(), ves.end());
	set_intersection(fes.begin(), fes.end(), ves.begin(), ves.end(), back_inserter(sharedes));
	assert(sharedes.size() == 2);//true for sure
	if (sharedes[0] == idx.edge) idx.edge = sharedes[1];else idx.edge = sharedes[0];

	return idx;
}
Index poly_fem::Navigation3D::switch_face(const Mesh &M, Index idx) {

	vector<uint32_t> efs = M.Es[idx.edge].neighbor_fs, hfs = M.Hs[idx.hedra].fs, sharedfs;
	sort(hfs.begin(), hfs.end()); sort(efs.begin(), efs.end());
	set_intersection(hfs.begin(), hfs.end(), efs.begin(), efs.end(), back_inserter(sharedfs));
	assert(sharedfs.size() == 2);//true for sure
	if (sharedfs[0] == idx.face) idx.face = sharedfs[1]; else idx.face = sharedfs[0];
	idx.face_corner = std::find(M.Fs[idx.face].vs.begin(), M.Fs[idx.face].vs.end(), idx.vertex) - M.Fs[idx.face].vs.begin();
	if (!M.Hs[idx.hedra].fs_flag[idx.hedra_patch]) idx.face_corner = M.Fs[idx.face].vs.size() - 1 - idx.face_corner;

	return idx;
}
Index poly_fem::Navigation3D::switch_hedra(const Mesh &M, Index idx) {

	if (M.Fs[idx.face].neighbor_hs.size() == 1)
		idx.hedra = -1;
	else {
		if (M.Fs[idx.face].neighbor_hs[0] == idx.hedra)
			idx.hedra = M.Fs[idx.face].neighbor_hs[1];
		else idx.hedra = M.Fs[idx.face].neighbor_hs[0];
	}
	idx.face_corner = std::find(M.Fs[idx.face].vs.begin(), M.Fs[idx.face].vs.end(), idx.vertex) - M.Fs[idx.face].vs.begin();
	if (!M.Hs[idx.hedra].fs_flag[idx.hedra_patch]) idx.face_corner = M.Fs[idx.face].vs.size() - 1 - idx.face_corner;

	return idx;
}
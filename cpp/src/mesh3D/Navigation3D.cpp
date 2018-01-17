#include "Navigation3D.hpp"
#include <algorithm>
#include <iterator>
#include <set>
#include <cassert>

using namespace poly_fem::Navigation3D;
using namespace poly_fem;
using namespace std;



void poly_fem::Navigation3D::prepare_mesh(Mesh3DStorage &M) {
	M.type = MeshType::Hyb;
	MeshProcessing3D::build_connectivity(M);
	MeshProcessing3D::global_orientation_hexes(M);
}


poly_fem::Navigation3D::Index poly_fem::Navigation3D::get_index_from_element_face(const Mesh3DStorage &M, int hi)
{
	Index idx;

	if (hi >= M.elements.size()) hi = hi % M.elements.size();
	if (M.elements[hi].hex) {
		idx.element = hi;

		vector<uint32_t> fvs, fvs_;
		fvs.insert(fvs.end(), M.elements[hi].vs.begin(), M.elements[hi].vs.begin() + 4);
		sort(fvs.begin(), fvs.end());
		idx.element_patch = -1;
		for (uint32_t i = 0; i < 6; i++) {
			idx.element_patch = i;
			fvs_ = M.faces[M.elements[hi].fs[i]].vs;
			sort(fvs_.begin(), fvs_.end());
			if (std::equal(fvs.begin(), fvs.end(), fvs_.begin())) break;
		}
		assert(idx.element_patch != -1);
		idx.face = M.elements[hi].fs[idx.element_patch];

		idx.vertex = M.elements[hi].vs[0];
		idx.face_corner = find(M.faces[idx.face].vs.begin(), M.faces[idx.face].vs.end(), idx.vertex) - M.faces[idx.face].vs.begin();

		int v0 = idx.vertex, v1 = M.elements[hi].vs[1];
		vector<uint32_t> ves0 = M.vertices[v0].neighbor_es, ves1 = M.vertices[v1].neighbor_es, sharedes;
		sort(ves0.begin(), ves0.end()); sort(ves1.begin(), ves1.end());
		set_intersection(ves0.begin(), ves0.end(), ves1.begin(), ves1.end(), back_inserter(sharedes));
		assert(sharedes.size() == 1);
		idx.edge = sharedes[0];
	}
	else {
		idx = get_index_from_element_face(M, hi, 0, 0);
	}

	return idx;
}

poly_fem::Navigation3D::Index poly_fem::Navigation3D::get_index_from_element_face(const Mesh3DStorage &M, int hi, int lf, int lv)
{
	Index idx;

	if (hi >= M.elements.size()) hi = hi % M.elements.size();
	idx.element = hi;

	if (lf >= M.elements[hi].fs.size()) lf = lf % M.elements[hi].fs.size();
	idx.element_patch = lf;
	idx.face = M.elements[hi].fs[idx.element_patch];

	if (lv >= M.faces[idx.face].vs.size()) lv = lv % M.faces[idx.face].vs.size();
	idx.face_corner = lv;
	idx.vertex = M.faces[idx.face].vs[idx.face_corner];
	//if (M.elements[hi].fs_flag[idx.element_patch]) idx.face_corner = M.faces[idx.face].vs.size() - 1 - idx.face_corner;
	if (M.elements[hi].fs_flag[idx.element_patch]) idx.face_corner = (idx.face_corner + M.faces[idx.face].vs.size() - 1)% M.faces[idx.face].vs.size();
	idx.edge = M.faces[idx.face].es[idx.face_corner];

	return idx;
}

// Navigation in a surface mesh
poly_fem::Navigation3D::Index poly_fem::Navigation3D::switch_vertex(const Mesh3DStorage &M, Index idx) {

	if(idx.vertex == M.edges[idx.edge].vs[0])idx.vertex = M.edges[idx.edge].vs[1];
	else idx.vertex = M.edges[idx.edge].vs[0];
	idx.face_corner = std::find(M.faces[idx.face].vs.begin(), M.faces[idx.face].vs.end(), idx.vertex) - M.faces[idx.face].vs.begin();
	//if (!M.elements[idx.element].fs_flag[idx.element_patch]) idx.face_corner = M.faces[idx.face].vs.size() - 1 - idx.face_corner;

	return idx;
}

poly_fem::Navigation3D::Index poly_fem::Navigation3D::switch_edge(const Mesh3DStorage &M, Index idx) {

	vector<uint32_t> ves = M.vertices[idx.vertex].neighbor_es, fes = M.faces[idx.face].es, sharedes;
	sort(fes.begin(), fes.end()); sort(ves.begin(), ves.end());
	set_intersection(fes.begin(), fes.end(), ves.begin(), ves.end(), back_inserter(sharedes));
	assert(sharedes.size() == 2);//true for sure
	if (sharedes[0] == idx.edge) idx.edge = sharedes[1];else idx.edge = sharedes[0];

	return idx;
}

poly_fem::Navigation3D::Index poly_fem::Navigation3D::switch_face(const Mesh3DStorage &M, Index idx) {

	vector<uint32_t> efs = M.edges[idx.edge].neighbor_fs, hfs = M.elements[idx.element].fs, sharedfs;
	sort(hfs.begin(), hfs.end()); sort(efs.begin(), efs.end());
	set_intersection(hfs.begin(), hfs.end(), efs.begin(), efs.end(), back_inserter(sharedfs));
	assert(sharedfs.size() == 2);//true for sure
	if (sharedfs[0] == idx.face) idx.face = sharedfs[1]; else idx.face = sharedfs[0];
	idx.face_corner = std::find(M.faces[idx.face].vs.begin(), M.faces[idx.face].vs.end(), idx.vertex) - M.faces[idx.face].vs.begin();
	if (!M.elements[idx.element].fs_flag[idx.element_patch]) idx.face_corner = M.faces[idx.face].vs.size() - 1 - idx.face_corner;

	return idx;
}

poly_fem::Navigation3D::Index poly_fem::Navigation3D::switch_element(const Mesh3DStorage &M, Index idx) {

	if (M.faces[idx.face].neighbor_hs.size() == 1) {
		idx.element = -1;
		return idx;
	}
	else {
		if (M.faces[idx.face].neighbor_hs[0] == idx.element)
			idx.element = M.faces[idx.face].neighbor_hs[1];
		else idx.element = M.faces[idx.face].neighbor_hs[0];

		idx.element_patch = find(M.elements[idx.element].fs.begin(), M.elements[idx.element].fs.end(), idx.face) - M.elements[idx.element].fs.begin();
	}
	idx.face_corner = std::find(M.faces[idx.face].vs.begin(), M.faces[idx.face].vs.end(), idx.vertex) - M.faces[idx.face].vs.begin();
	if (!M.elements[idx.element].fs_flag[idx.element_patch]) idx.face_corner = M.faces[idx.face].vs.size() - 1 - idx.face_corner;

	return idx;
}

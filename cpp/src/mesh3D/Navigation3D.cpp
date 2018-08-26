#include <polyfem/Navigation3D.hpp>
#include <igl/Timer.h>
#include <algorithm>
#include <iterator>
#include <set>
#include <cassert>

using namespace polyfem::Navigation3D;
using namespace polyfem;
using namespace std;


double polyfem::Navigation3D::get_index_from_element_face_time;
double polyfem::Navigation3D::switch_vertex_time;
double polyfem::Navigation3D::switch_edge_time;
double polyfem::Navigation3D::switch_face_time;
double polyfem::Navigation3D::switch_element_time;

void polyfem::Navigation3D::prepare_mesh(Mesh3DStorage &M) {
	if (M.type != MeshType::Tet)M.type = MeshType::Hyb;
	MeshProcessing3D::build_connectivity(M);
	MeshProcessing3D::global_orientation_hexes(M);
}


polyfem::Navigation3D::Index polyfem::Navigation3D::get_index_from_element_face(const Mesh3DStorage &M, int hi)
{
	igl::Timer timer; timer.start();

	Index idx;

	if (hi >= M.elements.size()) hi = hi % M.elements.size();
	if (M.elements[hi].hex) {
		 idx.element = hi;
		// idx.element_patch = 0;
		// idx.face = M.elements[hi].fs[idx.element_patch];

		// idx.vertex = M.elements[hi].vs[0];
		// idx.face_corner = 0;
		// idx.edge = M.faces[idx.face].es[0];		

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
		idx.face = M.elements[hi].fs[idx.element_patch];

		idx.vertex = M.elements[hi].vs[0];
		idx.face_corner = find(M.faces[idx.face].vs.begin(), M.faces[idx.face].vs.end(), idx.vertex) - M.faces[idx.face].vs.begin();

		int v0 = idx.vertex, v1 = M.elements[hi].vs[1];
		const vector<uint32_t> &ves0 = M.vertices[v0].neighbor_es, &ves1 = M.vertices[v1].neighbor_es;
		std::array<uint32_t, 2> sharedes;
		int num=1;
		MeshProcessing3D::set_intersection_own(ves0, ves1,sharedes, num);
		idx.edge = sharedes[0];
		get_index_from_element_face_time += timer.getElapsedTime();
	}
	else {
		idx = get_index_from_element_face(M, hi, 0, 0);
	}

	return idx;
}

polyfem::Navigation3D::Index polyfem::Navigation3D::get_index_from_element_face(const Mesh3DStorage &M, int hi, int lf, int lv)
{
	igl::Timer timer; timer.start();
	Index idx;

	if (hi >= M.elements.size()) hi = hi % M.elements.size();
	idx.element = hi;

	if (lf >= M.elements[hi].fs.size()) lf = lf % M.elements[hi].fs.size();
	idx.element_patch = lf;
	idx.face = M.elements[hi].fs[idx.element_patch];

	if (lv >= M.faces[idx.face].vs.size()) lv = lv % M.faces[idx.face].vs.size();
	idx.face_corner = lv;
	idx.vertex = M.faces[idx.face].vs[idx.face_corner];

	if (M.elements[hi].fs_flag[idx.element_patch]) idx.face_corner = (idx.face_corner + M.faces[idx.face].vs.size() - 1)% M.faces[idx.face].vs.size();
	idx.edge = M.faces[idx.face].es[idx.face_corner];

	//timer.stop();
	get_index_from_element_face_time += timer.getElapsedTime();

	return idx;
}

// Navigation in a surface mesh
polyfem::Navigation3D::Index polyfem::Navigation3D::switch_vertex(const Mesh3DStorage &M, Index idx) {
	igl::Timer timer; timer.start();
	if(idx.vertex == M.edges[idx.edge].vs[0])idx.vertex = M.edges[idx.edge].vs[1];
	else idx.vertex = M.edges[idx.edge].vs[0];

	int &corner = idx.face_corner, n = M.faces[idx.face].vs.size(), corner_1 = (corner-1+n)%n, corner1 = (corner+1)%n;
	if(M.faces[idx.face].vs[corner1] == idx.vertex) idx.face_corner = corner1;
	else if(M.faces[idx.face].vs[corner_1] == idx.vertex) idx.face_corner = corner_1;	
	//idx.face_corner = std::find(M.faces[idx.face].vs.begin(), M.faces[idx.face].vs.end(), idx.vertex) - M.faces[idx.face].vs.begin();

	//if (!M.elements[idx.element].fs_flag[idx.element_patch]) idx.face_corner = M.faces[idx.face].vs.size() - 1 - idx.face_corner;
	switch_vertex_time += timer.getElapsedTime();
	return idx;
}

polyfem::Navigation3D::Index polyfem::Navigation3D::switch_edge(const Mesh3DStorage &M, Index idx) {
	igl::Timer timer; timer.start();

	int n = M.faces[idx.face].vs.size();
	if(idx.edge == M.faces[idx.face].es[idx.face_corner]) idx.edge = M.faces[idx.face].es[(idx.face_corner-1+n)%n];
	else idx.edge = M.faces[idx.face].es[idx.face_corner];

	// const vector<uint32_t> &ves = M.vertices[idx.vertex].neighbor_es, &fes = M.faces[idx.face].es;
	// array<uint32_t, 2> sharedes;
	// int num=2;
	// MeshProcessing3D::set_intersection_own(ves, fes,sharedes, num);
	// assert(sharedes.size() == 2);//true for sure		assert(sharedes.size() == 2);//true for sure
	// if (sharedes[0] == idx.edge) idx.edge = sharedes[1];else idx.edge = sharedes[0];
	
	switch_edge_time += timer.getElapsedTime();
	return idx;
}

polyfem::Navigation3D::Index polyfem::Navigation3D::switch_face(const Mesh3DStorage &M, Index idx) {
	igl::Timer timer; timer.start();

	const vector<uint32_t> &efs = M.edges[idx.edge].neighbor_fs, &hfs = M.elements[idx.element].fs;
	std::array<uint32_t, 2> sharedfs;
	int num=2;
	MeshProcessing3D::set_intersection_own(efs, hfs,sharedfs, num);
	if (sharedfs[0] == idx.face) idx.face = sharedfs[1]; else idx.face = sharedfs[0];

	const vector<uint32_t> &fvs = M.faces[idx.face].vs;
	for(int i=0;i<fvs.size();i++) if(idx.vertex == fvs[i]){idx.face_corner=i; break;}
	//if (!M.elements[idx.element].fs_flag[idx.element_patch]) idx.face_corner = M.faces[idx.face].vs.size() - 1 - idx.face_corner;

	switch_face_time += timer.getElapsedTime();

	return idx;
}

polyfem::Navigation3D::Index polyfem::Navigation3D::switch_element(const Mesh3DStorage &M, Index idx) {
	igl::Timer timer; timer.start();
	if (M.faces[idx.face].neighbor_hs.size() == 1) {
		idx.element = -1;
		return idx;
	}
	else {
		if (M.faces[idx.face].neighbor_hs[0] == idx.element)
			idx.element = M.faces[idx.face].neighbor_hs[1];
		else idx.element = M.faces[idx.face].neighbor_hs[0];

		const vector<uint32_t> &fs = M.elements[idx.element].fs;
		for(int i=0;i<fs.size();i++) if(idx.face == fs[i]){idx.element_patch = i; break;}
		//idx.element_patch = find(M.elements[idx.element].fs.begin(), M.elements[idx.element].fs.end(), idx.face) - M.elements[idx.element].fs.begin();
	}
	const vector<uint32_t> &fvs = M.faces[idx.face].vs;
	for(int i=0;i<fvs.size();i++) if(idx.vertex == fvs[i]){idx.face_corner=i; break;}
	//idx.face_corner = std::find(M.faces[idx.face].vs.begin(), M.faces[idx.face].vs.end(), idx.vertex) - M.faces[idx.face].vs.begin();
	//if (!M.elements[idx.element].fs_flag[idx.element_patch]) idx.face_corner = M.faces[idx.face].vs.size() - 1 - idx.face_corner;

	switch_element_time += timer.getElapsedTime();
	return idx;
}

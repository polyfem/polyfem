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
	if(M.type == MeshType::Tet){
		idx.element = hi;
		idx.element_patch = 0;
		idx.face = M.HF(0, hi);
		idx.face_corner = 0;
		idx.vertex = M.FV(0, idx.face);
		idx.edge = M.FE(0, idx.face);

		get_index_from_element_face_time += timer.getElapsedTime();
	}
	else if (M.elements[hi].hex) {
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

	//if (M.elements[hi].fs_flag[idx.element_patch]) idx.face_corner = (idx.face_corner + M.faces[idx.face].vs.size() - 1)% M.faces[idx.face].vs.size();
	idx.edge = M.faces[idx.face].es[idx.face_corner];

	//timer.stop();
	get_index_from_element_face_time += timer.getElapsedTime();

	return idx;
}

// Navigation in a surface mesh
polyfem::Navigation3D::Index polyfem::Navigation3D::switch_vertex(const Mesh3DStorage &M, Index idx) {
	igl::Timer timer; timer.start();

	if(M.type == MeshType::Tet){
		if(idx.vertex == M.EV(0, idx.edge)) idx.vertex = M.EV(1, idx.edge);
		else idx.vertex = M.EV(0, idx.edge);

		if (M.FV(0, idx.face) == idx.vertex) idx.face_corner = 0;
		else if(M.FV(1, idx.face) == idx.vertex) idx.face_corner = 1;
		else idx.face_corner = 2;
	}		
	else{
		if(idx.vertex == M.edges[idx.edge].vs[0])idx.vertex = M.edges[idx.edge].vs[1];
		else idx.vertex = M.edges[idx.edge].vs[0];

		int &corner = idx.face_corner, n = M.faces[idx.face].vs.size(), corner_1 = (corner-1+n)%n, corner1 = (corner+1)%n;
		if(M.faces[idx.face].vs[corner1] == idx.vertex) idx.face_corner = corner1;
		else if(M.faces[idx.face].vs[corner_1] == idx.vertex) idx.face_corner = corner_1;	
	}	
	switch_vertex_time += timer.getElapsedTime();
	return idx;
}

polyfem::Navigation3D::Index polyfem::Navigation3D::switch_edge(const Mesh3DStorage &M, Index idx) {
	igl::Timer timer; timer.start();

	if(M.type == MeshType::Tet){
		if(idx.edge == M.FE(idx.face_corner, idx.face)) idx.edge = M.FE((idx.face_corner +2)%3,idx.face);
		else idx.edge = M.FE(idx.face_corner,idx.face);
	}else{
		int n = M.faces[idx.face].vs.size();
		if(idx.edge == M.faces[idx.face].es[idx.face_corner]) idx.edge = M.faces[idx.face].es[(idx.face_corner-1+n)%n];
		else idx.edge = M.faces[idx.face].es[idx.face_corner];
	}	
	switch_edge_time += timer.getElapsedTime();
	return idx;
}

polyfem::Navigation3D::Index polyfem::Navigation3D::switch_face(const Mesh3DStorage &M, Index idx) {
	igl::Timer timer; timer.start();
	if(M.type == MeshType::Tet){
		for(int i=0;i<4;i++){
			const auto & fid = M.HF(i,idx.element);
			if(fid == idx.face) continue;
			if(M.FE(0,fid) == idx.edge ||M.FE(1,fid) == idx.edge ||M.FE(2,fid) == idx.edge){
				idx.face = fid;
				idx.element_patch = i;
				if(M.FV(0, fid) == idx.vertex) idx.face_corner =0;
				else if(M.FV(1, fid) == idx.vertex) idx.face_corner =1;
				else idx.face_corner =2;
				break;
			}
		}
	}
	else{
		const vector<uint32_t> &efs = M.edges[idx.edge].neighbor_fs, &hfs = M.elements[idx.element].fs;
		std::array<uint32_t, 2> sharedfs;
		int num=2;
		MeshProcessing3D::set_intersection_own(efs, hfs,sharedfs, num);
		if (sharedfs[0] == idx.face) idx.face = sharedfs[1]; else idx.face = sharedfs[0];
		for(int i=0;i<hfs.size();i++) if(idx.face == hfs[i]){idx.element_patch=i; break;}

		const vector<uint32_t> &fvs = M.faces[idx.face].vs;
		for(int i=0;i<fvs.size();i++) if(idx.vertex == fvs[i]){idx.face_corner=i; break;}
	}

	switch_face_time += timer.getElapsedTime();

	return idx;
}

polyfem::Navigation3D::Index polyfem::Navigation3D::switch_element(const Mesh3DStorage &M, Index idx) {
	igl::Timer timer; timer.start();

	if(M.type == MeshType::Tet){
		if(M.FH(1, idx.face) == -1){
			idx.element = -1;
			return idx;
		}
		if(M.FH(0, idx.face) == idx.element){
			idx.element = M.FH(1, idx.face);
			idx.element_patch = M.FHi(1, idx.face);
		}
		else {
			idx.element = M.FH(0, idx.face);
			idx.element_patch = M.FHi(0, idx.face);
		}
	}
	else {
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
		}
		// const vector<uint32_t> &fvs = M.faces[idx.face].vs;
		// for(int i=0;i<fvs.size();i++) if(idx.vertex == fvs[i]){idx.face_corner=i; break;}
	}

	switch_element_time += timer.getElapsedTime();
	return idx;
}

#include "Navigation3D.hpp"
#include "MeshProcessing3D.hpp"

// #include <igl/Timer.h>
#include <algorithm>
#include <iterator>
#include <set>
#include <cassert>

using namespace polyfem::mesh::Navigation3D;
using namespace polyfem;
using namespace std;

// double polyfem::mesh::Navigation3D::get_index_from_element_face_time;
// double polyfem::mesh::Navigation3D::switch_vertex_time;
// double polyfem::mesh::Navigation3D::switch_edge_time;
// double polyfem::mesh::Navigation3D::switch_face_time;
// double polyfem::mesh::Navigation3D::switch_element_time;

void polyfem::mesh::Navigation3D::prepare_mesh(Mesh3DStorage &M)
{
	if (M.type != MeshType::TET)
		M.type = MeshType::HYB;
	MeshProcessing3D::build_connectivity(M);
	MeshProcessing3D::global_orientation_hexes(M);
}

polyfem::mesh::Navigation3D::Index polyfem::mesh::Navigation3D::get_index_from_element_face(const Mesh3DStorage &M, int hi)
{
	// igl::Timer timer; timer.start();

	Index idx;
	if (M.type == MeshType::TET)
	{
		idx.element = hi;
		idx.element_patch = 0;
		idx.face = M.HF(0, hi);
		idx.face_corner = 0;
		idx.vertex = M.FV(0, idx.face);
		idx.edge = M.FE(0, idx.face);

		if (M.elements[hi].fs_flag[idx.element_patch])
			idx.edge = M.FE(2, idx.face);
		// get_index_from_element_face_time += timer.getElapsedTime();
	}
	else if (M.elements[hi].hex)
	{
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

		for (uint32_t i = 0; i < 6; i++)
		{
			idx.element_patch = i;
			fvs_ = M.faces[M.elements[hi].fs[i]].vs;
			sort(fvs_.begin(), fvs_.end());
			if (std::equal(fvs.begin(), fvs.end(), fvs_.begin()))
				break;
		}
		idx.face = M.elements[hi].fs[idx.element_patch];

		idx.vertex = M.elements[hi].vs[0];
		idx.face_corner = find(M.faces[idx.face].vs.begin(), M.faces[idx.face].vs.end(), idx.vertex) - M.faces[idx.face].vs.begin();

		int v0 = idx.vertex, v1 = M.elements[hi].vs[1];
		const vector<uint32_t> &ves0 = M.vertices[v0].neighbor_es, &ves1 = M.vertices[v1].neighbor_es;
		std::array<uint32_t, 2> sharedes;
		int num = 1;
		MeshProcessing3D::set_intersection_own(ves0, ves1, sharedes, num);
		idx.edge = sharedes[0];
		// get_index_from_element_face_time += timer.getElapsedTime();
	}
	else
	{
		idx = get_index_from_element_face(M, hi, 0, 0);
	}

	return idx;
}

polyfem::mesh::Navigation3D::Index polyfem::mesh::Navigation3D::get_index_from_element_face(const Mesh3DStorage &M, int hi, int lf, int lv)
{
	// igl::Timer timer; timer.start();
	Index idx;

	if (hi >= M.elements.size())
		hi = hi % M.elements.size();
	idx.element = hi;

	if (lf >= M.elements[hi].fs.size())
		lf = lf % M.elements[hi].fs.size();
	idx.element_patch = lf;
	idx.face = M.elements[hi].fs[idx.element_patch];

	if (lv >= M.faces[idx.face].vs.size())
		lv = lv % M.faces[idx.face].vs.size();
	idx.face_corner = lv;
	idx.vertex = M.faces[idx.face].vs[idx.face_corner];

	int ei = idx.face_corner;
	if (M.elements[hi].fs_flag[idx.element_patch])
		ei = (idx.face_corner + M.faces[idx.face].vs.size() - 1) % M.faces[idx.face].vs.size();
	idx.edge = M.faces[idx.face].es[ei];
	// timer.stop();
	//  get_index_from_element_face_time += timer.getElapsedTime();

	return idx;
}
polyfem::mesh::Navigation3D::Index polyfem::mesh::Navigation3D::get_index_from_element_edge(const Mesh3DStorage &M, int hi, int v0i, int v1i)
{
	Index idx;
	idx.element = hi;
	idx.vertex = v0i;
	int v0 = v0i;
	int v1 = v1i;
	if (v0 > v1)
		std::swap(v0, v1);
	assert(v0 < v1);

	if (M.type == MeshType::TET)
	{
		for (int i = 0; i < 4; i++)
		{
			const auto &fid = M.HF(i, idx.element);
			for (int j = 0; j < 3; j++)
			{
				const auto &eid = M.FE(j, fid);
				assert(M.EV(0, eid) < M.EV(1, eid));
				if (M.EV(0, eid) == v0 && M.EV(1, eid) == v1)
				{
					idx.element_patch = i;
					idx.face = fid;
					idx.edge = eid;
					if (M.FV(0, fid) == idx.vertex)
						idx.face_corner = 0;
					else if (M.FV(1, fid) == idx.vertex)
						idx.face_corner = 1;
					else
						idx.face_corner = 2;

					assert(idx.vertex == v0i);
					assert(switch_vertex(M, idx).vertex == v1i);
					assert(idx.element == hi);

					return idx;
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < M.elements[hi].fs.size(); i++)
		{
			const auto &fid = M.elements[hi].fs[i];
			for (int j = 0; j < M.faces[fid].es.size(); j++)
			{
				const auto &eid = M.faces[fid].es[j];
				assert(M.edges[eid].vs[0] < M.edges[eid].vs[1]);
				if (M.edges[eid].vs[0] == v0 && M.edges[eid].vs[1] == v1)
				{
					idx.element_patch = i;
					idx.face = fid;
					idx.edge = eid;
					for (int k = 0; k < M.faces[fid].vs.size(); k++)
						if (M.faces[fid].vs[k] == idx.vertex)
							idx.face_corner = k;

					assert(idx.vertex == v0i);
					assert(switch_vertex(M, idx).vertex == v1i);
					assert(idx.element == hi);

					return idx;
				}
			}
		}
	}

	assert(false);
	return idx;
}

polyfem::mesh::Navigation3D::Index polyfem::mesh::Navigation3D::get_index_from_element_tri(const Mesh3DStorage &M, int hi, int v0i, int v1i, int v2i)
{
	int v0 = v0i;
	int v1 = v1i;
	int v2 = v2i;

	Index idx;
	idx.element = hi;
	idx.vertex = v0;
	int v0_ = v0, v1_ = v1;
	if (v0_ > v1_)
		swap(v0_, v1_);

	if (v0 > v2)
		swap(v0, v2);
	if (v0 > v1)
		swap(v0, v1);
	if (v1 > v2)
		swap(v1, v2);

	assert(v0 < v1);
	assert(v0 < v2);
	assert(v1 < v2);

	if (M.type == MeshType::TET)
	{
		for (int i = 0; i < 4; i++)
		{
			const auto &fid = M.HF(i, idx.element);
			int fv0 = M.FV(0, fid), fv1 = M.FV(1, fid), fv2 = M.FV(2, fid);
			if (fv0 > fv2)
				swap(fv0, fv2);
			if (fv0 > fv1)
				swap(fv0, fv1);
			if (fv1 > fv2)
				swap(fv1, fv2);

			assert(fv0 < fv1);
			assert(fv0 < fv2);
			assert(fv1 < fv2);

			if (v0 != fv0 || v1 != fv1 || v2 != fv2)
				continue;

			idx.face = fid;
			idx.element_patch = i;

			for (int j = 0; j < 3; j++)
			{
				const auto &eid = M.FE(j, fid);
				assert(M.EV(0, eid) < M.EV(1, eid));
				if (M.EV(0, eid) == v0_ && M.EV(1, eid) == v1_)
				{
					idx.edge = eid;
					if (M.FV(0, fid) == idx.vertex)
						idx.face_corner = 0;
					else if (M.FV(1, fid) == idx.vertex)
						idx.face_corner = 1;
					else
						idx.face_corner = 2;

					assert(idx.vertex == v0i);
					assert(switch_vertex(M, idx).vertex == v1i);
					assert(switch_vertex(M, switch_edge(M, idx)).vertex == v2i);
					return idx;
				}
			}
		}
	}
	else
	{
		assert(M.elements[idx.element].fs.size() == 4);
		for (int i = 0; i < 4; i++)
		{
			const auto fid = M.elements[idx.element].fs[i];
			const auto &fvid = M.faces[fid].vs;
			int fv0 = fvid[0], fv1 = fvid[1], fv2 = fvid[2];
			if (fv0 > fv2)
				swap(fv0, fv2);
			if (fv0 > fv1)
				swap(fv0, fv1);
			if (fv1 > fv2)
				swap(fv1, fv2);

			assert(fv0 < fv1);
			assert(fv0 < fv2);
			assert(fv1 < fv2);

			if (v0 != fv0 || v1 != fv1 || v2 != fv2)
				continue;

			idx.face = fid;
			idx.element_patch = i;

			for (int j = 0; j < 3; j++)
			{
				const auto eid = M.faces[fid].es[j];
				const auto &veid = M.edges[eid].vs;
				assert(veid[0] < veid[1]);
				if (veid[0] == v0_ && veid[1] == v1_)
				{
					idx.edge = eid;
					if (fvid[0] == idx.vertex)
						idx.face_corner = 0;
					else if (fvid[1] == idx.vertex)
						idx.face_corner = 1;
					else
						idx.face_corner = 2;

					assert(idx.vertex == v0i);
					assert(switch_vertex(M, idx).vertex == v1i);
					assert(switch_vertex(M, switch_edge(M, idx)).vertex == v2i);
					return idx;
				}
			}
		}
	}
	assert(false);
	return idx;
}
// Navigation in a surface mesh
polyfem::mesh::Navigation3D::Index polyfem::mesh::Navigation3D::switch_vertex(const Mesh3DStorage &M, Index idx)
{
	// igl::Timer timer; timer.start();

	if (M.type == MeshType::TET)
	{
		if (idx.vertex == M.EV(0, idx.edge))
			idx.vertex = M.EV(1, idx.edge);
		else
			idx.vertex = M.EV(0, idx.edge);

		if (M.FV(0, idx.face) == idx.vertex)
			idx.face_corner = 0;
		else if (M.FV(1, idx.face) == idx.vertex)
			idx.face_corner = 1;
		else
			idx.face_corner = 2;
	}
	else
	{
		if (idx.vertex == M.edges[idx.edge].vs[0])
			idx.vertex = M.edges[idx.edge].vs[1];
		else
			idx.vertex = M.edges[idx.edge].vs[0];

		int &corner = idx.face_corner, n = M.faces[idx.face].vs.size(), corner_1 = (corner - 1 + n) % n, corner1 = (corner + 1) % n;
		if (M.faces[idx.face].vs[corner1] == idx.vertex)
			idx.face_corner = corner1;
		else if (M.faces[idx.face].vs[corner_1] == idx.vertex)
			idx.face_corner = corner_1;
	}
	// switch_vertex_time += timer.getElapsedTime();
	return idx;
}

polyfem::mesh::Navigation3D::Index polyfem::mesh::Navigation3D::switch_edge(const Mesh3DStorage &M, Index idx)
{
	// igl::Timer timer; timer.start();

	if (M.type == MeshType::TET)
	{
		if (idx.edge == M.FE(idx.face_corner, idx.face))
			idx.edge = M.FE((idx.face_corner + 2) % 3, idx.face);
		else
			idx.edge = M.FE(idx.face_corner, idx.face);
	}
	else
	{
		int n = M.faces[idx.face].vs.size();
		if (idx.edge == M.faces[idx.face].es[idx.face_corner])
			idx.edge = M.faces[idx.face].es[(idx.face_corner - 1 + n) % n];
		else
			idx.edge = M.faces[idx.face].es[idx.face_corner];
	}
	// switch_edge_time += timer.getElapsedTime();
	return idx;
}

polyfem::mesh::Navigation3D::Index polyfem::mesh::Navigation3D::switch_face(const Mesh3DStorage &M, Index idx)
{
	// igl::Timer timer; timer.start();
	if (M.type == MeshType::TET)
	{
		for (int i = 0; i < 4; i++)
		{
			const auto &fid = M.HF(i, idx.element);
			if (fid == idx.face)
				continue;
			if (M.FE(0, fid) == idx.edge || M.FE(1, fid) == idx.edge || M.FE(2, fid) == idx.edge)
			{
				idx.face = fid;
				idx.element_patch = i;
				if (M.FV(0, fid) == idx.vertex)
					idx.face_corner = 0;
				else if (M.FV(1, fid) == idx.vertex)
					idx.face_corner = 1;
				else
					idx.face_corner = 2;
				break;
			}
		}
	}
	else
	{
		const vector<uint32_t> &efs = M.edges[idx.edge].neighbor_fs, &hfs = M.elements[idx.element].fs;
		std::array<uint32_t, 2> sharedfs;
		int num = 2;
		MeshProcessing3D::set_intersection_own(efs, hfs, sharedfs, num);
		if (sharedfs[0] == idx.face)
			idx.face = sharedfs[1];
		else
			idx.face = sharedfs[0];
		for (int i = 0; i < hfs.size(); i++)
			if (idx.face == hfs[i])
			{
				idx.element_patch = i;
				break;
			}

		const vector<uint32_t> &fvs = M.faces[idx.face].vs;
		for (int i = 0; i < fvs.size(); i++)
			if (idx.vertex == fvs[i])
			{
				idx.face_corner = i;
				break;
			}
	}

	// switch_face_time += timer.getElapsedTime();

	return idx;
}

polyfem::mesh::Navigation3D::Index polyfem::mesh::Navigation3D::switch_element(const Mesh3DStorage &M, Index idx)
{
	// igl::Timer timer; timer.start();

	if (M.type == MeshType::TET)
	{
		if (M.FH(1, idx.face) == -1)
		{
			idx.element = -1;
			return idx;
		}
		if (M.FH(0, idx.face) == idx.element)
		{
			idx.element = M.FH(1, idx.face);
			idx.element_patch = M.FHi(1, idx.face);
		}
		else
		{
			idx.element = M.FH(0, idx.face);
			idx.element_patch = M.FHi(0, idx.face);
		}
	}
	else
	{
		if (M.faces[idx.face].neighbor_hs.size() == 1)
		{
			idx.element = -1;
			return idx;
		}
		else
		{
			if (M.faces[idx.face].neighbor_hs[0] == idx.element)
				idx.element = M.faces[idx.face].neighbor_hs[1];
			else
				idx.element = M.faces[idx.face].neighbor_hs[0];

			const vector<uint32_t> &fs = M.elements[idx.element].fs;
			for (int i = 0; i < fs.size(); i++)
				if (idx.face == fs[i])
				{
					idx.element_patch = i;
					break;
				}
		}
		// const vector<uint32_t> &fvs = M.faces[idx.face].vs;
		// for(int i=0;i<fvs.size();i++) if(idx.vertex == fvs[i]){idx.face_corner=i; break;}
	}

	// switch_element_time += timer.getElapsedTime();
	return idx;
}

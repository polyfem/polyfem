#include "MeshProcessing3D.hpp"
#include <polyfem/utils/Logger.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <map>
#include <set>
#include <queue>
#include <iterator>
#include <cassert>

using namespace polyfem::mesh;
using namespace polyfem;
using namespace std;
using namespace Eigen;

void MeshProcessing3D::build_connectivity(Mesh3DStorage &hmi)
{
	hmi.edges.clear();
	if (hmi.type == MeshType::TRI || hmi.type == MeshType::QUA || hmi.type == MeshType::H_SUR)
	{
		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> temp;
		temp.reserve(hmi.faces.size() * 3);
		for (uint32_t i = 0; i < hmi.faces.size(); ++i)
		{
			int vn = hmi.faces[i].vs.size();
			for (uint32_t j = 0; j < vn; ++j)
			{
				uint32_t v0 = hmi.faces[i].vs[j], v1 = hmi.faces[i].vs[(j + 1) % vn];
				if (v0 > v1)
					std::swap(v0, v1);
				temp.push_back(std::make_tuple(v0, v1, i, j));
			}
			hmi.faces[i].es.resize(vn);
		}
		std::sort(temp.begin(), temp.end());
		hmi.edges.reserve(temp.size() / 2);
		uint32_t E_num = 0;
		Edge e;
		e.boundary = true;
		e.vs.resize(2);
		for (uint32_t i = 0; i < temp.size(); ++i)
		{
			if (i == 0 || (i != 0 && (std::get<0>(temp[i]) != std::get<0>(temp[i - 1]) || std::get<1>(temp[i]) != std::get<1>(temp[i - 1]))))
			{
				e.id = E_num;
				E_num++;
				e.vs[0] = std::get<0>(temp[i]);
				e.vs[1] = std::get<1>(temp[i]);
				hmi.edges.push_back(e);
			}
			else if (i != 0 && (std::get<0>(temp[i]) == std::get<0>(temp[i - 1]) && std::get<1>(temp[i]) == std::get<1>(temp[i - 1])))
				hmi.edges[E_num - 1].boundary = false;

			hmi.faces[std::get<2>(temp[i])].es[std::get<3>(temp[i])] = E_num - 1;
		}
		// boundary
		for (auto &v : hmi.vertices)
			v.boundary = false;
		for (uint32_t i = 0; i < hmi.edges.size(); ++i)
			if (hmi.edges[i].boundary)
			{
				hmi.vertices[hmi.edges[i].vs[0]].boundary = hmi.vertices[hmi.edges[i].vs[1]].boundary = true;
			}
	}
	else if (hmi.type == MeshType::HEX)
	{

		std::vector<std::vector<uint32_t>> total_fs(hmi.elements.size() * 6);
		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> tempF(hmi.elements.size() * 6);
		std::vector<uint32_t> vs(4);
		for (uint32_t i = 0; i < hmi.elements.size(); ++i)
		{
			for (short j = 0; j < 6; j++)
			{
				for (short k = 0; k < 4; k++)
					vs[k] = hmi.elements[i].vs[hex_face_table[j][k]];
				uint32_t id = 6 * i + j;
				total_fs[id] = vs;
				std::sort(vs.begin(), vs.end());
				tempF[id] = std::make_tuple(vs[0], vs[1], vs[2], vs[3], id, i, j);
			}
			hmi.elements[i].fs.resize(6);
		}
		std::sort(tempF.begin(), tempF.end());
		hmi.faces.reserve(tempF.size() / 3);
		Face f;
		f.boundary = true;
		uint32_t F_num = 0;
		for (uint32_t i = 0; i < tempF.size(); ++i)
		{
			if (i == 0 || (i != 0 && (std::get<0>(tempF[i]) != std::get<0>(tempF[i - 1]) || std::get<1>(tempF[i]) != std::get<1>(tempF[i - 1]) || std::get<2>(tempF[i]) != std::get<2>(tempF[i - 1]) || std::get<3>(tempF[i]) != std::get<3>(tempF[i - 1]))))
			{
				f.id = F_num;
				F_num++;
				f.vs = total_fs[std::get<4>(tempF[i])];
				hmi.faces.push_back(f);
			}
			else if (i != 0 && (std::get<0>(tempF[i]) == std::get<0>(tempF[i - 1]) && std::get<1>(tempF[i]) == std::get<1>(tempF[i - 1]) && std::get<2>(tempF[i]) == std::get<2>(tempF[i - 1]) && std::get<3>(tempF[i]) == std::get<3>(tempF[i - 1])))
				hmi.faces[F_num - 1].boundary = false;

			hmi.elements[std::get<5>(tempF[i])].fs[std::get<6>(tempF[i])] = F_num - 1;
		}

		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> temp(hmi.faces.size() * 4);
		for (uint32_t i = 0; i < hmi.faces.size(); ++i)
		{
			for (uint32_t j = 0; j < 4; ++j)
			{
				uint32_t v0 = hmi.faces[i].vs[j], v1 = hmi.faces[i].vs[(j + 1) % 4];
				if (v0 > v1)
					std::swap(v0, v1);
				temp[4 * i + j] = std::make_tuple(v0, v1, i, j);
			}
			hmi.faces[i].es.resize(4);
		}
		std::sort(temp.begin(), temp.end());
		hmi.edges.reserve(temp.size() / 2);
		uint32_t E_num = 0;
		Edge e;
		e.boundary = false;
		e.vs.resize(2);
		for (uint32_t i = 0; i < temp.size(); ++i)
		{
			if (i == 0 || (i != 0 && (std::get<0>(temp[i]) != std::get<0>(temp[i - 1]) || std::get<1>(temp[i]) != std::get<1>(temp[i - 1]))))
			{
				e.id = E_num;
				E_num++;
				e.vs[0] = std::get<0>(temp[i]);
				e.vs[1] = std::get<1>(temp[i]);
				hmi.edges.push_back(e);
			}
			hmi.faces[std::get<2>(temp[i])].es[std::get<3>(temp[i])] = E_num - 1;
		}
		// boundary
		for (auto &v : hmi.vertices)
			v.boundary = false;
		for (uint32_t i = 0; i < hmi.faces.size(); ++i)
			if (hmi.faces[i].boundary)
				for (uint32_t j = 0; j < 4; ++j)
				{
					uint32_t eid = hmi.faces[i].es[j];
					hmi.edges[eid].boundary = true;
					hmi.vertices[hmi.edges[eid].vs[0]].boundary = hmi.vertices[hmi.edges[eid].vs[1]].boundary = true;
				}
	}
	else if (hmi.type == MeshType::HYB || hmi.type == MeshType::TET)
	{
		vector<bool> bf_flag(hmi.faces.size(), false);
		for (auto h : hmi.elements)
			for (auto f : h.fs)
				bf_flag[f] = !bf_flag[f];
		for (auto &f : hmi.faces)
			f.boundary = bf_flag[f.id];

		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> temp;
		for (uint32_t i = 0; i < hmi.faces.size(); ++i)
		{
			int fl = hmi.faces[i].vs.size();
			for (uint32_t j = 0; j < hmi.faces[i].vs.size(); ++j)
			{
				uint32_t v0 = hmi.faces[i].vs[j], v1 = hmi.faces[i].vs[(j + 1) % fl];
				if (v0 > v1)
					std::swap(v0, v1);
				temp.push_back(std::make_tuple(v0, v1, i, j));
			}
			hmi.faces[i].es.resize(fl);
		}
		std::sort(temp.begin(), temp.end());
		hmi.edges.reserve(temp.size() / 2);
		uint32_t E_num = 0;
		Edge e;
		e.boundary = false;
		e.vs.resize(2);
		for (uint32_t i = 0; i < temp.size(); ++i)
		{
			if (i == 0 || (i != 0 && (std::get<0>(temp[i]) != std::get<0>(temp[i - 1]) || std::get<1>(temp[i]) != std::get<1>(temp[i - 1]))))
			{
				e.id = E_num;
				E_num++;
				e.vs[0] = std::get<0>(temp[i]);
				e.vs[1] = std::get<1>(temp[i]);
				hmi.edges.push_back(e);
			}
			hmi.faces[std::get<2>(temp[i])].es[std::get<3>(temp[i])] = E_num - 1;
		}
		// boundary
		for (auto &v : hmi.vertices)
			v.boundary = false;
		for (uint32_t i = 0; i < hmi.faces.size(); ++i)
			if (hmi.faces[i].boundary)
				for (uint32_t j = 0; j < hmi.faces[i].vs.size(); ++j)
				{
					uint32_t eid = hmi.faces[i].es[j];
					hmi.edges[eid].boundary = true;
					hmi.vertices[hmi.faces[i].vs[j]].boundary = true;
				}
	}
	// f_nhs;
	for (auto &f : hmi.faces)
		f.neighbor_hs.clear();
	for (uint32_t i = 0; i < hmi.elements.size(); i++)
	{
		for (uint32_t j = 0; j < hmi.elements[i].fs.size(); j++)
		{
			hmi.faces[hmi.elements[i].fs[j]].neighbor_hs.push_back(i);
		}
	}
	// e_nfs, v_nfs
	for (auto &e : hmi.edges)
		e.neighbor_fs.clear();
	for (auto &v : hmi.vertices)
		v.neighbor_fs.clear();
	for (uint32_t i = 0; i < hmi.faces.size(); i++)
	{
		for (uint32_t j = 0; j < hmi.faces[i].es.size(); j++)
			hmi.edges[hmi.faces[i].es[j]].neighbor_fs.push_back(i);
		for (uint32_t j = 0; j < hmi.faces[i].vs.size(); j++)
			hmi.vertices[hmi.faces[i].vs[j]].neighbor_fs.push_back(i);
	}
	// v_nes, v_nvs
	for (auto &v : hmi.vertices)
	{
		v.neighbor_es.clear();
		v.neighbor_vs.clear();
	}
	for (uint32_t i = 0; i < hmi.edges.size(); i++)
	{
		uint32_t v0 = hmi.edges[i].vs[0], v1 = hmi.edges[i].vs[1];
		hmi.vertices[v0].neighbor_es.push_back(i);
		hmi.vertices[v1].neighbor_es.push_back(i);
		hmi.vertices[v0].neighbor_vs.push_back(v1);
		hmi.vertices[v1].neighbor_vs.push_back(v0);
	}
	// e_nhs
	for (auto &e : hmi.edges)
		e.neighbor_hs.clear();
	for (auto &ele : hmi.elements)
		ele.es.clear();
	for (uint32_t i = 0; i < hmi.edges.size(); i++)
	{
		std::vector<uint32_t> nhs;
		for (uint32_t j = 0; j < hmi.edges[i].neighbor_fs.size(); j++)
		{
			uint32_t nfid = hmi.edges[i].neighbor_fs[j];
			nhs.insert(nhs.end(), hmi.faces[nfid].neighbor_hs.begin(), hmi.faces[nfid].neighbor_hs.end());
		}
		std::sort(nhs.begin(), nhs.end());
		nhs.erase(std::unique(nhs.begin(), nhs.end()), nhs.end());
		hmi.edges[i].neighbor_hs = nhs;
		for (auto nhid : nhs)
			hmi.elements[nhid].es.push_back(i);
	}
	// v_nhs; ordering fs for hex
	if (hmi.type != MeshType::HYB && hmi.type != MeshType::TET)
		return;

	for (auto &v : hmi.vertices)
		v.neighbor_hs.clear();
	for (uint32_t i = 0; i < hmi.elements.size(); i++)
	{
		vector<uint32_t> vs;
		for (auto fid : hmi.elements[i].fs)
			vs.insert(vs.end(), hmi.faces[fid].vs.begin(), hmi.faces[fid].vs.end());
		sort(vs.begin(), vs.end());
		vs.erase(unique(vs.begin(), vs.end()), vs.end());

		bool degree3 = true;
		for (auto vid : vs)
		{
			int nv = 0;
			for (auto nvid : hmi.vertices[vid].neighbor_vs)
				if (find(vs.begin(), vs.end(), nvid) != vs.end())
					nv++;
			if (nv != 3)
			{
				degree3 = false;
				break;
			}
		}

		if (hmi.elements[i].hex && (vs.size() != 8 || !degree3))
			hmi.elements[i].hex = false;

		hmi.elements[i].vs.clear();

		if (hmi.elements[i].hex)
		{
			int top_fid = hmi.elements[i].fs[0];
			hmi.elements[i].vs = hmi.faces[top_fid].vs;

			std::set<uint32_t> s_model(vs.begin(), vs.end());
			std::set<uint32_t> s_pattern(hmi.faces[top_fid].vs.begin(), hmi.faces[top_fid].vs.end());
			vector<uint32_t> vs_left;
			std::set_difference(s_model.begin(), s_model.end(), s_pattern.begin(), s_pattern.end(), std::back_inserter(vs_left));

			for (auto vid : hmi.faces[top_fid].vs)
				for (auto nvid : hmi.vertices[vid].neighbor_vs)
					if (find(vs_left.begin(), vs_left.end(), nvid) != vs_left.end())
					{
						hmi.elements[i].vs.push_back(nvid);
						break;
					}

			function<int(vector<uint32_t> &, int &)> WHICH_F = [&](vector<uint32_t> &vs0, int &f_flag) -> int {
				int which_f = -1;
				sort(vs0.begin(), vs0.end());
				bool found_f = false;
				for (uint32_t j = 0; j < hmi.elements[i].fs.size(); j++)
				{
					auto fid = hmi.elements[i].fs[j];
					vector<uint32_t> vs1 = hmi.faces[fid].vs;
					sort(vs1.begin(), vs1.end());
					if (vs0.size() == vs1.size() && std::equal(vs0.begin(), vs0.end(), vs1.begin()))
					{
						f_flag = hmi.elements[i].fs_flag[j];
						which_f = fid;
						break;
					}
				}
				return which_f;
			};

			vector<uint32_t> fs;
			vector<bool> fs_flag;
			fs_flag.push_back(hmi.elements[i].fs_flag[0]);
			fs.push_back(top_fid);
			vector<uint32_t> vs_temp;

			vs_temp.insert(vs_temp.end(), hmi.elements[i].vs.begin() + 4, hmi.elements[i].vs.end());
			int f_flag = -1;
			int bottom_fid = WHICH_F(vs_temp, f_flag);
			fs_flag.push_back(f_flag);
			fs.push_back(bottom_fid);

			vs_temp.clear();
			vs_temp.push_back(hmi.elements[i].vs[0]);
			vs_temp.push_back(hmi.elements[i].vs[1]);
			vs_temp.push_back(hmi.elements[i].vs[4]);
			vs_temp.push_back(hmi.elements[i].vs[5]);
			f_flag = -1;
			int front_fid = WHICH_F(vs_temp, f_flag);
			fs_flag.push_back(f_flag);
			fs.push_back(front_fid);

			vs_temp.clear();
			vs_temp.push_back(hmi.elements[i].vs[2]);
			vs_temp.push_back(hmi.elements[i].vs[3]);
			vs_temp.push_back(hmi.elements[i].vs[6]);
			vs_temp.push_back(hmi.elements[i].vs[7]);
			f_flag = -1;
			int back_fid = WHICH_F(vs_temp, f_flag);
			fs_flag.push_back(f_flag);
			fs.push_back(back_fid);

			vs_temp.clear();
			vs_temp.push_back(hmi.elements[i].vs[1]);
			vs_temp.push_back(hmi.elements[i].vs[2]);
			vs_temp.push_back(hmi.elements[i].vs[5]);
			vs_temp.push_back(hmi.elements[i].vs[6]);
			f_flag = -1;
			int left_fid = WHICH_F(vs_temp, f_flag);
			fs_flag.push_back(f_flag);
			fs.push_back(left_fid);

			vs_temp.clear();
			vs_temp.push_back(hmi.elements[i].vs[3]);
			vs_temp.push_back(hmi.elements[i].vs[0]);
			vs_temp.push_back(hmi.elements[i].vs[7]);
			vs_temp.push_back(hmi.elements[i].vs[4]);
			f_flag = -1;
			int right_fid = WHICH_F(vs_temp, f_flag);
			fs_flag.push_back(f_flag);
			fs.push_back(right_fid);

			hmi.elements[i].fs = fs;
			hmi.elements[i].fs_flag = fs_flag;
		}
		else
			hmi.elements[i].vs = vs;
		for (uint32_t j = 0; j < hmi.elements[i].vs.size(); j++)
			hmi.vertices[hmi.elements[i].vs[j]].neighbor_hs.push_back(i);
	}
	// matrix representation of tet mesh
	if (hmi.type == MeshType::TET)
	{
		hmi.EV.resize(2, hmi.edges.size());
		for (const auto &e : hmi.edges)
		{
			hmi.EV(0, e.id) = e.vs[0];
			hmi.EV(1, e.id) = e.vs[1];
		}
		hmi.FV.resize(3, hmi.faces.size());
		hmi.FE.resize(3, hmi.faces.size());
		hmi.FH.resize(2, hmi.faces.size());
		hmi.FHi.resize(2, hmi.faces.size());
		for (const auto &f : hmi.faces)
		{
			hmi.FV(0, f.id) = f.vs[0];
			hmi.FV(1, f.id) = f.vs[1];
			hmi.FV(2, f.id) = f.vs[2];

			hmi.FE(0, f.id) = f.es[0];
			hmi.FE(1, f.id) = f.es[1];
			hmi.FE(2, f.id) = f.es[2];

			hmi.FH(0, f.id) = f.neighbor_hs[0];
			for (int i = 0; i < hmi.elements[f.neighbor_hs[0]].fs.size(); i++)
				if (f.id == hmi.elements[f.neighbor_hs[0]].fs[i])
					hmi.FHi(0, f.id) = i;

			hmi.FH(1, f.id) = -1;
			hmi.FHi(1, f.id) = -1;
			if (f.neighbor_hs.size() == 2)
			{
				hmi.FH(1, f.id) = f.neighbor_hs[1];
				for (int i = 0; i < hmi.elements[f.neighbor_hs[1]].fs.size(); i++)
					if (f.id == hmi.elements[f.neighbor_hs[1]].fs[i])
						hmi.FHi(1, f.id) = i;
			}
		}
		hmi.HV.resize(4, hmi.elements.size());
		hmi.HF.resize(4, hmi.elements.size());
		for (const auto &h : hmi.elements)
		{
			hmi.HV(0, h.id) = h.vs[0];
			hmi.HV(1, h.id) = h.vs[1];
			hmi.HV(2, h.id) = h.vs[2];
			hmi.HV(3, h.id) = h.vs[3];

			hmi.HF(0, h.id) = h.fs[0];
			hmi.HF(1, h.id) = h.fs[1];
			hmi.HF(2, h.id) = h.fs[2];
			hmi.HF(3, h.id) = h.fs[3];
		}
	}

	// boundary flags for hybrid mesh
	std::vector<bool> bv_flag(hmi.vertices.size(), false), be_flag(hmi.edges.size(), false), bf_flag(hmi.faces.size(), false);
	for (auto f : hmi.faces)
		if (f.boundary && hmi.elements[f.neighbor_hs[0]].hex)
			bf_flag[f.id] = true;
		else if (!f.boundary)
		{
			int ele0 = f.neighbor_hs[0], ele1 = f.neighbor_hs[1];
			if ((hmi.elements[ele0].hex && !hmi.elements[ele1].hex) || (!hmi.elements[ele0].hex && hmi.elements[ele1].hex))
				bf_flag[f.id] = true;
		}
	for (uint32_t i = 0; i < hmi.faces.size(); ++i)
		if (bf_flag[i])
			for (uint32_t j = 0; j < hmi.faces[i].vs.size(); ++j)
			{
				uint32_t eid = hmi.faces[i].es[j];
				be_flag[eid] = true;
				bv_flag[hmi.faces[i].vs[j]] = true;
			}
	// boundary_hex for hybrid mesh
	for (auto &v : hmi.vertices)
		v.boundary_hex = false;
	for (auto &e : hmi.edges)
		e.boundary_hex = false;
	for (auto &f : hmi.faces)
		f.boundary_hex = bf_flag[f.id];
	for (auto &ele : hmi.elements)
		if (ele.hex)
		{
			for (auto vid : ele.vs)
			{
				if (!bv_flag[vid])
					continue;
				int fn = 0;
				for (auto nfid : hmi.vertices[vid].neighbor_fs)
					if (bf_flag[nfid] && std::find(ele.fs.begin(), ele.fs.end(), nfid) != ele.fs.end())
						fn++;
				if (fn == 3)
					hmi.vertices[vid].boundary_hex = true;
			}
			for (auto eid : ele.es)
			{
				if (!be_flag[eid])
					continue;
				int fn = 0;
				for (auto nfid : hmi.edges[eid].neighbor_fs)
					if (bf_flag[nfid] && std::find(ele.fs.begin(), ele.fs.end(), nfid) != ele.fs.end())
						fn++;
				if (fn == 2)
					hmi.edges[eid].boundary_hex = true;
			}
		}
}
void MeshProcessing3D::reorder_hex_mesh_propogation(Mesh3DStorage &hmi)
{
	// connected components
	vector<bool> H_tag(hmi.elements.size(), false);
	vector<vector<uint32_t>> Groups;
	while (true)
	{
		vector<uint32_t> group;
		int shid = -1;
		for (uint32_t i = 0; i < H_tag.size(); i++)
			if (!H_tag[i])
			{
				shid = i;
				break;
			}
		if (shid == -1)
			break;
		group.push_back(shid);
		H_tag[shid] = true;
		vector<uint32_t> group_ = group;
		while (true)
		{
			vector<uint32_t> pool;
			for (auto hid : group_)
			{
				vector<vector<uint32_t>> Fvs(6), fvs_sorted;
				for (uint32_t i = 0; i < 6; i++)
					for (uint32_t j = 0; j < 4; j++)
						Fvs[i].push_back(hmi.elements[hid].vs[hex_face_table[i][j]]);
				fvs_sorted = Fvs;
				for (auto &vs : fvs_sorted)
					sort(vs.begin(), vs.end());

				for (auto fid : hmi.elements[hid].fs)
					if (!hmi.faces[fid].boundary)
					{
						int nhid = hmi.faces[fid].neighbor_hs[0];
						if (nhid == hid)
							nhid = hmi.faces[fid].neighbor_hs[1];

						if (!H_tag[nhid])
						{
							pool.push_back(nhid);
							H_tag[nhid] = true;

							vector<uint32_t> fvs = hmi.faces[fid].vs;
							sort(fvs.begin(), fvs.end());

							int f_ind = -1;
							for (uint32_t i = 0; i < 6; i++)
								if (std::equal(fvs.begin(), fvs.end(), fvs_sorted[i].begin()))
								{
									f_ind = i;
									break;
								}
							vector<uint32_t> hvs = hmi.elements[nhid].vs;
							hmi.elements[nhid].vs.clear();
							vector<uint32_t> topvs = Fvs[f_ind];
							std::reverse(topvs.begin(), topvs.end());
							vector<uint32_t> bottomvs;
							for (uint32_t i = 0; i < 4; i++)
							{
								for (auto nvid : hmi.vertices[topvs[i]].neighbor_vs)
									if (nvid != topvs[(i + 3) % 4] && nvid != topvs[(i + 1) % 4]
										&& std::find(hvs.begin(), hvs.end(), nvid) != hvs.end())
									{
										bottomvs.push_back(nvid);
										break;
									}
							}
							hmi.elements[nhid].vs = topvs;
							hmi.elements[nhid].vs.insert(hmi.elements[nhid].vs.end(), bottomvs.begin(), bottomvs.end());
						}
					}
			}
			if (pool.size())
			{
				group_ = pool;
				group.insert(group.end(), pool.begin(), pool.end());
				pool.size();
			}
			else
				break;
		}
		Groups.push_back(group);
	}
	// direction
	for (auto group : Groups)
	{
		Mesh_Quality mq1, mq2;
		Mesh3DStorage m1, m2;
		m2.type = m1.type = MeshType::HEX;

		m2.points = m1.points = hmi.points;
		for (auto hid : group)
			m1.elements.push_back(hmi.elements[hid]);
		scaled_jacobian(m1, mq1);
		if (mq1.min_Jacobian > 0)
			continue;
		m2.elements = m1.elements;
		for (auto &h : m2.elements)
		{
			swap(h.vs[1], h.vs[3]);
			swap(h.vs[5], h.vs[7]);
		}
		scaled_jacobian(m2, mq2);
		if (mq2.ave_Jacobian > mq1.ave_Jacobian)
		{
			for (auto &h : m2.elements)
				hmi.elements[h.id] = h;
		}
	}
}
bool MeshProcessing3D::scaled_jacobian(Mesh3DStorage &hmi, Mesh_Quality &mq)
{
	if (hmi.type != MeshType::HEX)
		return false;

	mq.ave_Jacobian = 0;
	mq.min_Jacobian = 1;
	mq.deviation_Jacobian = 0;
	mq.V_Js.resize(hmi.elements.size() * 8);
	mq.V_Js.setZero();
	mq.H_Js.resize(hmi.elements.size());
	mq.H_Js.setZero();

	for (uint32_t i = 0; i < hmi.elements.size(); i++)
	{
		double hex_minJ = 1;
		for (uint32_t j = 0; j < 8; j++)
		{
			uint32_t v0, v1, v2, v3;
			v0 = hex_tetra_table[j][0];
			v1 = hex_tetra_table[j][1];
			v2 = hex_tetra_table[j][2];
			v3 = hex_tetra_table[j][3];

			Vector3d c0 = hmi.points.col(hmi.elements[i].vs[v0]);
			Vector3d c1 = hmi.points.col(hmi.elements[i].vs[v1]);
			Vector3d c2 = hmi.points.col(hmi.elements[i].vs[v2]);
			Vector3d c3 = hmi.points.col(hmi.elements[i].vs[v3]);

			double jacobian_value = a_jacobian(c0, c1, c2, c3);

			if (hex_minJ > jacobian_value)
				hex_minJ = jacobian_value;

			uint32_t id = 8 * i + j;
			mq.V_Js[id] = jacobian_value;
		}
		mq.H_Js[i] = hex_minJ;
		mq.ave_Jacobian += hex_minJ;
		if (mq.min_Jacobian > hex_minJ)
			mq.min_Jacobian = hex_minJ;
	}
	mq.ave_Jacobian /= hmi.elements.size();
	for (int i = 0; i < mq.H_Js.size(); i++)
		mq.deviation_Jacobian += (mq.H_Js[i] - mq.ave_Jacobian) * (mq.H_Js[i] - mq.ave_Jacobian);
	mq.deviation_Jacobian /= hmi.elements.size();

	return true;
}
double MeshProcessing3D::a_jacobian(Vector3d &v0, Vector3d &v1, Vector3d &v2, Vector3d &v3)
{
	Matrix3d Jacobian;

	Jacobian.col(0) = v1 - v0;
	Jacobian.col(1) = v2 - v0;
	Jacobian.col(2) = v3 - v0;

	double norm1 = Jacobian.col(0).norm();
	double norm2 = Jacobian.col(1).norm();
	double norm3 = Jacobian.col(2).norm();

	double scaled_jacobian = Jacobian.determinant();
	if (std::abs(norm1) < Jacobian_Precision || std::abs(norm2) < Jacobian_Precision || std::abs(norm3) < Jacobian_Precision)
	{
		return scaled_jacobian;
	}
	scaled_jacobian /= norm1 * norm2 * norm3;
	return scaled_jacobian;
}

void MeshProcessing3D::global_orientation_hexes(Mesh3DStorage &hmi)
{
	Mesh3DStorage mesh;
	mesh.type = MeshType::HEX;
	mesh.points = hmi.points;
	mesh.vertices.resize(hmi.vertices.size());
	for (const auto &v : hmi.vertices)
	{
		Vertex v_;
		v_.id = v.id;
		v_.v = v.v;
		mesh.vertices[v.id] = v;
	}
	vector<int> Ele_map(hmi.elements.size(), -1), Ele_map_reverse;
	for (const auto &ele : hmi.elements)
	{
		if (!ele.hex)
			continue;
		Element ele_;
		ele_.id = mesh.elements.size();
		ele_.vs = ele.vs;
		for (auto vid : ele_.vs)
			mesh.vertices[vid].neighbor_hs.push_back(ele_.id);
		mesh.elements.push_back(ele_);

		Ele_map[ele.id] = ele_.id;
		Ele_map_reverse.push_back(ele.id);
	}

	build_connectivity(mesh);
	reorder_hex_mesh_propogation(mesh);

	for (const auto &ele : mesh.elements)
		hmi.elements[Ele_map_reverse[ele.id]].vs = ele.vs;
}
void MeshProcessing3D::refine_catmul_clark_polar(Mesh3DStorage &M, int iter, bool reverse, std::vector<int> &Parents)
{

	for (int i = 0; i < iter; i++)
	{

		std::vector<int> Refinement_Levels;
		ele_subdivison_levels(M, Refinement_Levels);

		Mesh3DStorage M_;
		M_.type = MeshType::HYB;

		vector<int> E2V(M.edges.size()), F2V(M.faces.size()), Ele2V(M.elements.size());

		int vn = 0;
		for (auto v : M.vertices)
		{
			Vertex v_;
			v_.id = vn++;
			v_.v.resize(3);
			for (int j = 0; j < 3; j++)
				v_.v[j] = M.points(j, v.id);
			M_.vertices.push_back(v_);
		}

		for (auto e : M.edges)
		{
			Vertex v;
			v.id = vn++;
			v.v.resize(3);

			Vector3d center;
			center.setZero();
			for (auto vid : e.vs)
				center += M.points.col(vid);
			center /= e.vs.size();
			for (int j = 0; j < 3; j++)
				v.v[j] = center[j];

			M_.vertices.push_back(v);
			E2V[e.id] = v.id;
		}
		for (auto f : M.faces)
		{
			Vertex v;
			v.id = vn++;
			v.v.resize(3);

			Vector3d center;
			center.setZero();
			for (auto vid : f.vs)
				center += M.points.col(vid);
			center /= f.vs.size();
			for (int j = 0; j < 3; j++)
				v.v[j] = center[j];

			M_.vertices.push_back(v);
			F2V[f.id] = v.id;
		}
		for (auto ele : M.elements)
		{
			if (!ele.hex)
				continue;
			Vertex v;
			v.id = vn++;
			v.v = ele.v_in_Kernel;

			M_.vertices.push_back(v);
			Ele2V[ele.id] = v.id;
		}
		// new elements
		std::vector<std::vector<uint32_t>> total_fs;
		total_fs.reserve(M.elements.size() * 8 * 6);
		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> tempF;
		tempF.reserve(M.elements.size() * 8 * 6);
		std::vector<uint32_t> vs(4);

		int elen = 0, fn = 0;
		for (auto ele : M.elements)
		{
			if (ele.hex)
			{
				for (auto vid : ele.vs)
				{
					// top 4 vs
					vector<int> top_vs(4);
					top_vs[0] = vid;
					int fid = -1;
					for (auto nfid : M.vertices[vid].neighbor_fs)
						if (find(ele.fs.begin(), ele.fs.end(), nfid) != ele.fs.end())
						{
							fid = nfid;
							break;
						}
					assert(fid != -1);
					top_vs[2] = F2V[fid];

					int v_ind = find(M.faces[fid].vs.begin(), M.faces[fid].vs.end(), vid) - M.faces[fid].vs.begin();
					int e_pre = M.faces[fid].es[(v_ind - 1 + 4) % 4];
					int e_aft = M.faces[fid].es[v_ind];
					top_vs[1] = E2V[e_pre];
					top_vs[3] = E2V[e_aft];
					// bottom 4 vs
					vector<int> bottom_vs(4);

					auto nvs = M.vertices[vid].neighbor_vs;
					int e_per = -1;
					for (auto nvid : nvs)
						if (find(ele.vs.begin(), ele.vs.end(), nvid) != ele.vs.end())
						{
							if (nvid != M.edges[e_pre].vs[0] && nvid != M.edges[e_pre].vs[1] && nvid != M.edges[e_aft].vs[0] && nvid != M.edges[e_aft].vs[1])
							{
								vector<uint32_t> sharedes, es0 = M.vertices[vid].neighbor_es, es1 = M.vertices[nvid].neighbor_es;
								sort(es0.begin(), es0.end());
								sort(es1.begin(), es1.end());
								set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(sharedes));
								assert(sharedes.size());
								e_per = sharedes[0];
								break;
							}
						}

					assert(e_per != -1);
					bottom_vs[0] = E2V[e_per];
					bottom_vs[2] = Ele2V[ele.id];

					int f_pre = -1;
					vector<uint32_t> sharedfs, fs0 = M.edges[e_pre].neighbor_fs, fs1 = M.edges[e_per].neighbor_fs;
					sort(fs0.begin(), fs0.end());
					sort(fs1.begin(), fs1.end());
					set_intersection(fs0.begin(), fs0.end(), fs1.begin(), fs1.end(), back_inserter(sharedfs));
					for (auto sfid : sharedfs)
						if (find(ele.fs.begin(), ele.fs.end(), sfid) != ele.fs.end())
						{
							f_pre = sfid;
							break;
						}
					assert(f_pre != -1);

					int f_aft = -1;
					sharedfs.clear();
					fs0 = M.edges[e_aft].neighbor_fs;
					fs1 = M.edges[e_per].neighbor_fs;
					sort(fs0.begin(), fs0.end());
					sort(fs1.begin(), fs1.end());
					set_intersection(fs0.begin(), fs0.end(), fs1.begin(), fs1.end(), back_inserter(sharedfs));
					for (auto sfid : sharedfs)
						if (find(ele.fs.begin(), ele.fs.end(), sfid) != ele.fs.end())
						{
							f_aft = sfid;
							break;
						}
					assert(f_aft != -1);

					bottom_vs[1] = F2V[f_pre];
					bottom_vs[3] = F2V[f_aft];

					vector<int> ele_vs = top_vs;
					ele_vs.insert(ele_vs.end(), bottom_vs.begin(), bottom_vs.end());
					// fs
					for (short j = 0; j < 6; j++)
					{
						for (short k = 0; k < 4; k++)
							vs[k] = ele_vs[hex_face_table[j][k]];
						total_fs.push_back(vs);
						std::sort(vs.begin(), vs.end());
						tempF.push_back(std::make_tuple(vs[0], vs[1], vs[2], vs[3], fn++, elen, j));
					}
					// new ele
					Element ele_;
					ele_.id = elen++;
					ele_.fs.resize(6, -1);
					ele_.fs_flag.resize(6, 1);

					ele_.hex = true;
					ele_.v_in_Kernel.resize(3);

					Vector3d center;
					center.setZero();
					for (auto evid : ele_vs)
						for (int j = 0; j < 3; j++)
							center[j] += M_.vertices[evid].v[j];
					center /= ele_vs.size();
					for (int j = 0; j < 3; j++)
						ele_.v_in_Kernel[j] = center[j];

					M_.elements.push_back(ele_);
					Parents.push_back(ele.id);
				}
			}
			else
			{
				int level = Refinement_Levels[ele.id];
				if (reverse)
					level = 1;
				// local_V2V
				std::vector<std::vector<int>> local_V2Vs;
				std::map<int, int> local_vi_map;
				for (auto vid : ele.vs)
				{
					std::vector<int> v2v;
					v2v.push_back(vid);
					for (int r = 0; r < level; r++)
					{
						Vertex v_;
						v_.id = vn++;
						v_.v.resize(3);

						for (int j = 0; j < 3; j++)
							v_.v[j] = M_.vertices[vid].v[j] + (ele.v_in_Kernel[j] - M_.vertices[vid].v[j]) * (r + 1.0) / (double)(level + 1);
						if (reverse)
						{
							for (int j = 0; j < 3; j++)
								v_.v[j] = M_.vertices[vid].v[j] + (M_.vertices[vid].v[j] - ele.v_in_Kernel[j]) * (r + 1.0) / (double)(level + 1);
						}
						M_.vertices.push_back(v_);
						v2v.push_back(v_.id);
					}
					local_vi_map[vid] = local_V2Vs.size();
					local_V2Vs.push_back(v2v);
				}
				// local_E2V
				vector<uint32_t> es;
				for (auto fid : ele.fs)
					es.insert(es.end(), M.faces[fid].es.begin(), M.faces[fid].es.end());
				sort(es.begin(), es.end());
				es.erase(unique(es.begin(), es.end()), es.end());

				std::vector<std::vector<int>> local_E2Vs;
				std::map<int, int> local_ei_map;
				for (auto eid : es)
				{
					std::vector<int> e2v;
					e2v.push_back(E2V[eid]);
					for (int r = 0; r < level; r++)
					{
						Vertex v_;
						v_.id = vn++;
						v_.v.resize(3);

						Vector3d center;
						center.setZero();
						for (auto vid : M.edges[eid].vs)
							for (int j = 0; j < 3; j++)
								center[j] += M_.vertices[local_V2Vs[local_vi_map[vid]][r + 1]].v[j];
						center /= M.edges[eid].vs.size();
						for (int j = 0; j < 3; j++)
							v_.v[j] = center[j];

						M_.vertices.push_back(v_);
						e2v.push_back(v_.id);
					}
					local_ei_map[eid] = local_E2Vs.size();
					local_E2Vs.push_back(e2v);
				}
				// local_F2V
				std::vector<std::vector<int>> local_F2Vs;
				std::map<int, int> local_fi_map;
				for (auto fid : ele.fs)
				{
					std::vector<int> f2v;
					f2v.push_back(F2V[fid]);
					for (int r = 0; r < level; r++)
					{
						Vertex v_;
						v_.id = vn++;
						v_.v.resize(3);

						Vector3d center;
						center.setZero();
						for (auto vid : M.faces[fid].vs)
							for (int j = 0; j < 3; j++)
								center[j] += M_.vertices[local_V2Vs[local_vi_map[vid]][r + 1]].v[j];
						center /= M.faces[fid].vs.size();
						for (int j = 0; j < 3; j++)
							v_.v[j] = center[j];

						M_.vertices.push_back(v_);
						f2v.push_back(v_.id);
					}
					local_fi_map[fid] = local_F2Vs.size();
					local_F2Vs.push_back(f2v);
				}
				// polyhedron fs
				int local_fn = 0;
				for (auto fid : ele.fs)
				{
					auto &fvs = M.faces[fid].vs;
					auto &fes = M.faces[fid].es;
					int fvn = M.faces[fid].vs.size();
					for (uint32_t j = 0; j < fvn; j++)
					{
						vs[0] = local_E2Vs[local_ei_map[fes[(j - 1 + fvn) % fvn]]][level];
						vs[1] = local_V2Vs[local_vi_map[fvs[j]]][level];
						vs[2] = local_E2Vs[local_ei_map[fes[j]]][level];
						vs[3] = local_F2Vs[local_fi_map[fid]][level];

						if (reverse)
						{
							vs[0] = local_E2Vs[local_ei_map[fes[(j - 1 + fvn) % fvn]]][0];
							vs[1] = local_V2Vs[local_vi_map[fvs[j]]][0];
							vs[2] = local_E2Vs[local_ei_map[fes[j]]][0];
							vs[3] = local_F2Vs[local_fi_map[fid]][0];
						}

						total_fs.push_back(vs);
						std::sort(vs.begin(), vs.end());
						tempF.push_back(std::make_tuple(vs[0], vs[1], vs[2], vs[3], fn++, elen, local_fn++));
					}
				}
				// polyhedron
				Element ele_;
				ele_.id = elen++;
				ele_.fs.resize(local_fn, -1);
				ele_.fs_flag.resize(local_fn, 1);

				ele_.hex = false;
				ele_.v_in_Kernel = ele.v_in_Kernel;
				M_.elements.push_back(ele_);
				Parents.push_back(ele.id);
				// hex
				for (int r = 0; r < level; r++)
				{
					for (auto fid : ele.fs)
					{
						auto &fvs = M.faces[fid].vs;
						auto &fes = M.faces[fid].es;
						int fvn = M.faces[fid].vs.size();
						for (uint32_t j = 0; j < fvn; j++)
						{
							vector<int> ele_vs(8);
							ele_vs[0] = local_E2Vs[local_ei_map[fes[(j - 1 + fvn) % fvn]]][r + 1];
							ele_vs[1] = local_V2Vs[local_vi_map[fvs[j]]][r + 1];
							ele_vs[2] = local_E2Vs[local_ei_map[fes[j]]][r + 1];
							ele_vs[3] = local_F2Vs[local_fi_map[fid]][r + 1];

							ele_vs[4] = local_E2Vs[local_ei_map[fes[(j - 1 + fvn) % fvn]]][r];
							ele_vs[5] = local_V2Vs[local_vi_map[fvs[j]]][r];
							ele_vs[6] = local_E2Vs[local_ei_map[fes[j]]][r];
							ele_vs[7] = local_F2Vs[local_fi_map[fid]][r];

							// fs
							for (short j = 0; j < 6; j++)
							{
								for (short k = 0; k < 4; k++)
									vs[k] = ele_vs[hex_face_table[j][k]];
								total_fs.push_back(vs);
								std::sort(vs.begin(), vs.end());
								tempF.push_back(std::make_tuple(vs[0], vs[1], vs[2], vs[3], fn++, elen, j));
							}
							// hex
							Element ele_;
							ele_.id = elen++;
							ele_.fs.resize(6, -1);
							ele_.fs_flag.resize(6, 1);
							ele_.hex = true;
							ele_.v_in_Kernel.resize(3);

							Vector3d center;
							center.setZero();
							for (auto vid : ele_vs)
								for (int j = 0; j < 3; j++)
									center[j] += M_.vertices[vid].v[j];
							center /= ele_vs.size();
							for (int j = 0; j < 3; j++)
								ele_.v_in_Kernel[j] = center[j];
							M_.elements.push_back(ele_);
							Parents.push_back(ele.id);
						}
					}
				}
			}
		}
		// Fs
		std::sort(tempF.begin(), tempF.end());
		M_.faces.reserve(tempF.size() / 3);
		Face f;
		f.boundary = true;
		uint32_t F_num = 0;
		for (uint32_t i = 0; i < tempF.size(); ++i)
		{
			if (i == 0 || (i != 0 && (std::get<0>(tempF[i]) != std::get<0>(tempF[i - 1]) || std::get<1>(tempF[i]) != std::get<1>(tempF[i - 1]) || std::get<2>(tempF[i]) != std::get<2>(tempF[i - 1]) || std::get<3>(tempF[i]) != std::get<3>(tempF[i - 1]))))
			{
				f.id = F_num;
				F_num++;
				f.vs = total_fs[std::get<4>(tempF[i])];
				M_.faces.push_back(f);
			}
			else if (i != 0 && (std::get<0>(tempF[i]) == std::get<0>(tempF[i - 1]) && std::get<1>(tempF[i]) == std::get<1>(tempF[i - 1]) && std::get<2>(tempF[i]) == std::get<2>(tempF[i - 1]) && std::get<3>(tempF[i]) == std::get<3>(tempF[i - 1])))
				M_.faces[F_num - 1].boundary = false;

			M_.elements[std::get<5>(tempF[i])].fs[std::get<6>(tempF[i])] = F_num - 1;
		}

		M_.points.resize(3, M_.vertices.size());
		for (auto v : M_.vertices)
			for (int j = 0; j < 3; j++)
				M_.points(j, v.id) = v.v[j];

		build_connectivity(M_);
		orient_volume_mesh(M_);
		build_connectivity(M_);

		M = M_;
	}
}
void MeshProcessing3D::refine_red_refinement_tet(Mesh3DStorage &M, int iter)
{
	for (int i = 0; i < iter; i++)
	{

		Mesh3DStorage M_;
		M_.type = MeshType::TET;

		vector<int> E2V(M.edges.size());

		int vn = 0;
		for (const auto &v : M.vertices)
		{
			Vertex v_;
			v_.id = vn++;
			v_.v.resize(3);
			for (int j = 0; j < 3; j++)
				v_.v[j] = M.points(j, v.id);
			M_.vertices.push_back(v_);
		}

		for (const auto &e : M.edges)
		{
			Vertex v;
			v.id = vn++;
			v.v.resize(3);

			Vector3d center;
			center.setZero();
			for (auto vid : e.vs)
				center += M.points.col(vid);
			center /= e.vs.size();
			for (int j = 0; j < 3; j++)
				v.v[j] = center[j];

			M_.vertices.push_back(v);
			E2V[e.id] = v.id;
		}
		// for (const auto &ele : M.elements) {
		// 	Element ele_;
		// 	ele_.id = M_.elements.size();
		// 	ele_.fs.resize(4, -1);
		// 	ele_.fs_flag.resize(4, 1);
		// 	ele_.vs = ele.vs;

		// 	ele_.hex = false;
		// 	ele_.v_in_Kernel.resize(3);

		// 	Vector3d center;
		// 	center.setZero();
		// 	for (const auto &evid : ele_.vs) for (int j = 0; j < 3; j++)center[j] += M_.vertices[evid].v[j];
		// 	center /= ele_.vs.size();
		// 	for (int j = 0; j < 3; j++)ele_.v_in_Kernel[j] = center[j];

		// 	M_.elements.push_back(ele_);
		// }

		auto shared_edge = [&](int v0, int v1, int &e) -> bool {
			auto &es0 = M.vertices[v0].neighbor_es, &es1 = M.vertices[v1].neighbor_es;
			std::sort(es0.begin(), es0.end());
			std::sort(es1.begin(), es1.end());
			std::vector<uint32_t> es;
			std::set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(es));
			if (es.size())
			{
				e = es[0];
				return true;
			}
			return false;
		};

		std::vector<bool> e_flag(M.edges.size(), false);

		for (const auto &ele : M.elements)
		{ // 1 --> 8

			for (short i = 0; i < 4; i++)
			{ // four corners
				Element ele_;
				ele_.id = M_.elements.size();
				ele_.fs.resize(4, -1);
				ele_.fs_flag.resize(4, 1);

				ele_.hex = false;
				ele_.v_in_Kernel.resize(3);

				ele_.vs.push_back(ele.vs[i]);
				for (short j = 0; j < 4; j++)
				{
					if (j == i)
						continue;
					int v0 = ele.vs[i], v1 = ele.vs[j];
					int e = -1;
					if (shared_edge(v0, v1, e))
						ele_.vs.push_back(E2V[e]);
				}
				Vector3d center;
				center.setZero();
				for (const auto &evid : ele_.vs)
					for (int j = 0; j < 3; j++)
						center[j] += M_.vertices[evid].v[j];
				center /= ele_.vs.size();
				for (int j = 0; j < 3; j++)
					ele_.v_in_Kernel[j] = center[j];

				M_.elements.push_back(ele_);
			}

			// 6 edges
			std::vector<int> edges(6);
			for (int i = 0; i < 6; i++)
			{
				int v0 = ele.vs[tet_edges[i][0]], v1 = ele.vs[tet_edges[i][1]];
				int e = -1;
				if (shared_edge(v0, v1, e))
					edges[i] = e;
			}
			// the longest edge
			int lv0 = E2V[edges[0]], lv1 = E2V[edges[5]];
			e_flag[edges[0]] = true;
			e_flag[edges[5]] = true;
			for (short i = 0; i < 4; i++)
			{ // four faces
				Element ele_;
				ele_.id = M_.elements.size();
				ele_.fs.resize(4, -1);
				ele_.fs_flag.resize(4, 1);

				ele_.hex = false;
				ele_.v_in_Kernel.resize(3);

				ele_.vs.push_back(lv0);
				ele_.vs.push_back(lv1);
				for (short j = 0; j < 3; j++)
				{
					int c_e = M.faces[ele.fs[i]].es[j];
					if (e_flag[c_e])
						continue;
					ele_.vs.push_back(E2V[c_e]);
				}
				Vector3d center;
				center.setZero();
				for (const auto &evid : ele_.vs)
					for (int j = 0; j < 3; j++)
						center[j] += M_.vertices[evid].v[j];
				center /= ele_.vs.size();
				for (int j = 0; j < 3; j++)
					ele_.v_in_Kernel[j] = center[j];

				M_.elements.push_back(ele_);
			}
			e_flag[edges[0]] = e_flag[edges[5]] = false;
		}

		M_.points.resize(3, M_.vertices.size());
		for (auto v : M_.vertices)
			for (int j = 0; j < 3; j++)
				M_.points(j, v.id) = v.v[j];

		// orient tets
		const auto &t = M.elements[0].vs;
		Vector3d c0 = M.points.col(t[0]);
		Vector3d c1 = M.points.col(t[1]);
		Vector3d c2 = M.points.col(t[2]);
		Vector3d c3 = M.points.col(t[3]);
		bool signed_volume = a_jacobian(c0, c1, c2, c3) > 0 ? true : false;

		for (auto &ele : M_.elements)
		{
			c0 = M_.points.col(ele.vs[0]);
			c1 = M_.points.col(ele.vs[1]);
			c2 = M_.points.col(ele.vs[2]);
			c3 = M_.points.col(ele.vs[3]);
			bool sign = a_jacobian(c0, c1, c2, c3) > 0 ? true : false;
			if (sign != signed_volume)
				std::swap(ele.vs[1], ele.vs[3]);
		}

		// Fs
		std::vector<std::vector<uint32_t>> total_fs;
		total_fs.reserve(M.elements.size() * 4);
		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> tempF;
		tempF.reserve(M.elements.size() * 4);
		std::vector<uint32_t> vs(3);
		for (const auto &ele : M_.elements)
		{
			for (int i = 0; i < 4; i++)
			{
				vs[0] = ele.vs[tet_faces[i][0]];
				vs[1] = ele.vs[tet_faces[i][1]];
				vs[2] = ele.vs[tet_faces[i][2]];
				total_fs.push_back(vs);
				std::sort(vs.begin(), vs.end());
				tempF.push_back(std::make_tuple(vs[0], vs[1], vs[2], ele.id * 4 + i, ele.id, i));
			}
		}
		std::sort(tempF.begin(), tempF.end());
		M_.faces.reserve(tempF.size() / 3);
		Face f;
		f.boundary = true;
		uint32_t F_num = 0;
		for (uint32_t i = 0; i < tempF.size(); ++i)
		{
			if (i == 0 || (i != 0 && (std::get<0>(tempF[i]) != std::get<0>(tempF[i - 1]) || std::get<1>(tempF[i]) != std::get<1>(tempF[i - 1]) || std::get<2>(tempF[i]) != std::get<2>(tempF[i - 1]))))
			{
				f.id = F_num;
				F_num++;
				f.vs = total_fs[std::get<3>(tempF[i])];
				M_.faces.push_back(f);
			}
			else if (i != 0 && (std::get<0>(tempF[i]) == std::get<0>(tempF[i - 1]) && std::get<1>(tempF[i]) == std::get<1>(tempF[i - 1]) && std::get<2>(tempF[i]) == std::get<2>(tempF[i - 1])))
				M_.faces[F_num - 1].boundary = false;

			M_.elements[std::get<4>(tempF[i])].fs[std::get<5>(tempF[i])] = F_num - 1;
		}

		build_connectivity(M_);
		orient_volume_mesh(M_);
		build_connectivity(M_);

		M = M_;

		double hmin = 10000, hmax = 0, havg = 0;
		for (const auto &e : M.edges)
		{
			Eigen::Vector3d v0 = M.points.col(e.vs[0]), v1 = M.points.col(e.vs[1]);
			double len = (v0 - v1).norm();
			if (len < hmin)
				hmin = len;
			if (len > hmax)
				hmax = len;
			havg += len;
		}
		havg /= M.edges.size();
	}
}
void MeshProcessing3D::straight_sweeping(const Mesh3DStorage &Mi, int sweep_coord, double height, int nlayer, Mesh3DStorage &Mo)
{
	if (sweep_coord > 2 || sweep_coord < 0)
	{
		logger().error("invalid sweeping direction!");
		return;
	}
	if (Mi.type != MeshType::H_SUR && Mi.type != MeshType::TRI && Mi.type != MeshType::QUA)
	{
		logger().error("invalid planar surface!");
		return;
	}
	if (height <= 0 || nlayer < 1)
	{
		logger().error("invalid height or number of layers!");
		return;
	}

	Mo.points.resize(3, 0);
	Mo.vertices.clear();
	Mo.edges.clear();
	Mo.faces.clear();
	Mo.elements.clear();
	Mo.type = MeshType::HYB;
	// v, layers
	std::vector<std::vector<int>> Vlayers(nlayer + 1);
	std::vector<double> interval(3, 0);
	interval[sweep_coord] = height / nlayer;

	for (int i = 0; i < nlayer + 1; i++)
	{
		std::vector<int> a_layer;
		for (auto v : Mi.vertices)
		{
			Vertex v_;
			v_.id = Mo.vertices.size();
			v_.v.resize(3);
			for (int j = 0; j < 3; j++)
				v_.v[j] = v.v[j] + i * interval[j];

			Mo.vertices.push_back(v_);
			a_layer.push_back(v_.id);
		}
		Vlayers[i] = a_layer;
	}
	// f
	std::vector<std::vector<int>> Flayers(nlayer + 1);
	for (int i = 0; i < nlayer + 1; i++)
	{
		std::vector<int> a_layer;
		for (auto f : Mi.faces)
		{
			Face f_;
			f_.id = Mo.faces.size();
			for (auto vid : f.vs)
				f_.vs.push_back(Vlayers[i][vid]);

			Mo.faces.push_back(f_);
			a_layer.push_back(f_.id);
		}
		Flayers[i] = a_layer;
	}
	// ef
	std::vector<std::vector<int>> EFlayers(nlayer);
	for (int i = 0; i < nlayer; i++)
	{
		std::vector<int> a_layer;
		for (auto e : Mi.edges)
		{
			Face f_;
			f_.id = Mo.faces.size();
			int v0 = e.vs[0], v1 = e.vs[1];
			f_.vs.push_back(Vlayers[i][v0]);
			f_.vs.push_back(Vlayers[i][v1]);
			f_.vs.push_back(Vlayers[i + 1][v1]);
			f_.vs.push_back(Vlayers[i + 1][v0]);

			Mo.faces.push_back(f_);
			a_layer.push_back(f_.id);
		}
		EFlayers[i] = a_layer;
	}
	// ele
	for (int i = 0; i < nlayer; i++)
	{
		for (auto f : Mi.faces)
		{
			Element ele;
			ele.id = Mo.elements.size();
			ele.hex = (f.vs.size() == 4);

			ele.fs.push_back(Flayers[i][f.id]);
			ele.fs.push_back(Flayers[i + 1][f.id]);
			for (auto eid : f.es)
				ele.fs.push_back(EFlayers[i][eid]);

			ele.fs_flag.resize(ele.fs.size(), false);

			ele.v_in_Kernel.resize(3, 0);
			int nv = 0;
			for (int j = 0; j < 2; j++)
			{
				nv += Mo.faces[ele.fs[j]].vs.size();
				for (auto vid : Mo.faces[ele.fs[j]].vs)
					for (int k = 0; k < 3; k++)
						ele.v_in_Kernel[k] += Mo.vertices[vid].v[k];
			}
			for (int k = 0; k < 3; k++)
				ele.v_in_Kernel[k] /= nv;

			Mo.elements.push_back(ele);
		}
	}

	Mo.points.resize(3, Mo.vertices.size());
	for (auto v : Mo.vertices)
		for (int j = 0; j < 3; j++)
			Mo.points(j, v.id) = v.v[j];

	build_connectivity(Mo);
	orient_volume_mesh(Mo);
	build_connectivity(Mo);
}

void MeshProcessing3D::orient_surface_mesh(Mesh3DStorage &hmi)
{

	vector<bool> flag(hmi.faces.size(), true);
	flag[0] = false;

	std::queue<uint32_t> pf_temp;
	pf_temp.push(0);
	while (!pf_temp.empty())
	{
		uint32_t fid = pf_temp.front();
		pf_temp.pop();
		for (auto eid : hmi.faces[fid].es)
			for (auto nfid : hmi.edges[eid].neighbor_fs)
			{
				if (!flag[nfid])
					continue;
				uint32_t v0 = hmi.edges[eid].vs[0], v1 = hmi.edges[eid].vs[1];
				int32_t v0_pos = std::find(hmi.faces[fid].vs.begin(), hmi.faces[fid].vs.end(), v0) - hmi.faces[fid].vs.begin();
				int32_t v1_pos = std::find(hmi.faces[fid].vs.begin(), hmi.faces[fid].vs.end(), v1) - hmi.faces[fid].vs.begin();

				if ((v0_pos + 1) % hmi.faces[fid].vs.size() != v1_pos)
					swap(v0, v1);

				int32_t v0_pos_ = std::find(hmi.faces[nfid].vs.begin(), hmi.faces[nfid].vs.end(), v0) - hmi.faces[nfid].vs.begin();
				int32_t v1_pos_ = std::find(hmi.faces[nfid].vs.begin(), hmi.faces[nfid].vs.end(), v1) - hmi.faces[nfid].vs.begin();

				if ((v0_pos_ + 1) % hmi.faces[nfid].vs.size() == v1_pos_)
					std::reverse(hmi.faces[nfid].vs.begin(), hmi.faces[nfid].vs.end());

				pf_temp.push(nfid);
				flag[nfid] = false;
			}
	}
	double res = 0;
	Vector3d ori;
	ori.setZero();
	for (auto f : hmi.faces)
	{
		auto &fvs = f.vs;
		Vector3d center;
		center.setZero();
		for (auto vid : fvs)
			center += hmi.points.col(vid);
		center /= fvs.size();

		for (uint32_t j = 0; j < fvs.size(); j++)
		{
			Vector3d x = hmi.points.col(fvs[j]) - ori, y = hmi.points.col(fvs[(j + 1) % fvs.size()]) - ori, z = center - ori;
			res += -((x[0] * y[1] * z[2] + x[1] * y[2] * z[0] + x[2] * y[0] * z[1]) - (x[2] * y[1] * z[0] + x[1] * y[0] * z[2] + x[0] * y[2] * z[1]));
		}
	}
	if (res > 0)
	{
		for (uint32_t i = 0; i < hmi.faces.size(); i++)
			std::reverse(hmi.faces[i].vs.begin(), hmi.faces[i].vs.end());
	}
}
void MeshProcessing3D::orient_volume_mesh(Mesh3DStorage &hmi)
{
	// surface orienting
	Mesh3DStorage M_sur;
	M_sur.type = MeshType::H_SUR;
	int bvn = 0;
	for (auto v : hmi.vertices)
		if (v.boundary)
			bvn++;
	M_sur.points.resize(3, bvn);
	bvn = 0;
	vector<int> V_map(hmi.vertices.size(), -1), V_map_reverse;
	for (auto v : hmi.vertices)
		if (v.boundary)
		{
			M_sur.points.col(bvn++) = hmi.points.col(v.id);
			Vertex v_;
			v_.id = M_sur.vertices.size();
			M_sur.vertices.push_back(v_);

			V_map[v.id] = v_.id;
			V_map_reverse.push_back(v.id);
		}
	for (auto f : hmi.faces)
		if (f.boundary)
		{
			for (int j = 0; j < f.vs.size(); j++)
			{
				f.id = M_sur.faces.size();
				f.es.clear();
				f.neighbor_hs.clear();
				f.vs[j] = V_map[f.vs[j]];
				M_sur.vertices[f.vs[j]].neighbor_fs.push_back(f.id);
			}
			M_sur.faces.push_back(f);
		}
	build_connectivity(M_sur);
	orient_surface_mesh(M_sur);

	int fn_ = 0;
	for (auto &f : hmi.faces)
		if (f.boundary)
		{
			for (int j = 0; j < f.vs.size(); j++)
				f.vs[j] = V_map_reverse[M_sur.faces[fn_].vs[j]];
			fn_++;
		}
	// volume orienting
	vector<bool> F_tag(hmi.faces.size(), true);
	std::vector<short> F_visit(hmi.faces.size(), 0); // 0 un-visited, 1 visited once, 2 visited twice
	for (uint32_t j = 0; j < hmi.faces.size(); j++)
		if (hmi.faces[j].boundary)
		{
			F_visit[j]++;
		}
	std::vector<bool> F_state(hmi.faces.size(), false); // false is the reverse direction, true is the same direction
	std::vector<bool> P_visit(hmi.elements.size(), false);
	while (true)
	{
		std::vector<uint32_t> candidates;
		for (uint32_t j = 0; j < F_visit.size(); j++)
			if (F_visit[j] == 1)
				candidates.push_back(j);
		if (!candidates.size())
			break;
		for (auto ca : candidates)
		{
			if (F_visit[ca] == 2)
				continue;
			uint32_t pid = hmi.faces[ca].neighbor_hs[0];
			if (P_visit[pid])
				if (hmi.faces[ca].neighbor_hs.size() == 2)
					pid = hmi.faces[ca].neighbor_hs[1];
			if (P_visit[pid])
			{
				logger().error("bug");
			}
			auto &fs = hmi.elements[pid].fs;
			for (auto fid : fs)
				F_tag[fid] = false;

			uint32_t start_f = ca;
			F_tag[start_f] = true;
			F_visit[ca]++;
			if (F_state[ca])
				F_state[ca] = false;
			else
				F_state[ca] = true;

			std::queue<uint32_t> pf_temp;
			pf_temp.push(start_f);
			while (!pf_temp.empty())
			{
				uint32_t fid = pf_temp.front();
				pf_temp.pop();
				for (auto eid : hmi.faces[fid].es)
					for (auto nfid : hmi.edges[eid].neighbor_fs)
					{

						if (F_tag[nfid])
							continue;
						uint32_t v0 = hmi.edges[eid].vs[0], v1 = hmi.edges[eid].vs[1];
						int32_t v0_pos = std::find(hmi.faces[fid].vs.begin(), hmi.faces[fid].vs.end(), v0) - hmi.faces[fid].vs.begin();
						int32_t v1_pos = std::find(hmi.faces[fid].vs.begin(), hmi.faces[fid].vs.end(), v1) - hmi.faces[fid].vs.begin();

						if ((v0_pos + 1) % hmi.faces[fid].vs.size() != v1_pos)
							std::swap(v0, v1);

						int32_t v0_pos_ = std::find(hmi.faces[nfid].vs.begin(), hmi.faces[nfid].vs.end(), v0) - hmi.faces[nfid].vs.begin();
						int32_t v1_pos_ = std::find(hmi.faces[nfid].vs.begin(), hmi.faces[nfid].vs.end(), v1) - hmi.faces[nfid].vs.begin();

						if (F_state[fid])
						{
							if ((v0_pos_ + 1) % hmi.faces[nfid].vs.size() == v1_pos_)
								F_state[nfid] = false;
							else
								F_state[nfid] = true;
						}
						else if (!F_state[fid])
						{
							if ((v0_pos_ + 1) % hmi.faces[nfid].vs.size() == v1_pos_)
								F_state[nfid] = true;
							else
								F_state[nfid] = false;
						}

						F_visit[nfid]++;

						pf_temp.push(nfid);
						F_tag[nfid] = true;
					}
			}
			P_visit[pid] = true;
			for (uint32_t j = 0; j < fs.size(); j++)
				hmi.elements[pid].fs_flag[j] = F_state[fs[j]];
		}
	}
}
void MeshProcessing3D::ele_subdivison_levels(const Mesh3DStorage &hmi, std::vector<int> &Ls)
{

	Ls.clear();
	Ls.resize(hmi.elements.size(), 1);
	std::vector<double> volumes(hmi.elements.size(), 0);

	auto compute_volume = [&](const int id, double &vol) {
		Vector3d ori;
		ori.setZero();
		for (auto f : hmi.elements[id].fs)
		{
			auto &fvs = hmi.faces[f].vs;
			Vector3d center;
			center.setZero();
			for (auto vid : fvs)
				center += hmi.points.col(vid);
			center /= fvs.size();

			for (uint32_t j = 0; j < fvs.size(); j++)
			{
				Vector3d x = hmi.points.col(fvs[j]) - ori, y = hmi.points.col(fvs[(j + 1) % fvs.size()]) - ori, z = center - ori;
				vol += -((x[0] * y[1] * z[2] + x[1] * y[2] * z[0] + x[2] * y[0] * z[1]) - (x[2] * y[1] * z[0] + x[1] * y[0] * z[2] + x[0] * y[2] * z[1]));
			}
		}
		vol = std::abs(vol);
	};

	for (const auto &ele : hmi.elements)
		compute_volume(ele.id, volumes[ele.id]);

	double ave_volume = 0;
	for (const auto &v : volumes)
		ave_volume += v;
	ave_volume /= volumes.size();
	for (int i = 0; i < Ls.size(); i++)
		if (!hmi.elements[i].hex)
		{
			Ls[i] = volumes[i] / ave_volume;
			if (Ls[i] < 1)
				Ls[i] = 1;
		}
}

// template<typename T>
// void MeshProcessing3D::set_intersection_own(const std::vector<T> &A, const std::vector<T> &B, std::vector<T> &C, const int &num){
void MeshProcessing3D::set_intersection_own(const std::vector<uint32_t> &A, const std::vector<uint32_t> &B, std::array<uint32_t, 2> &C, int &num)
{
	// void MeshProcessing3D::set_intersection_own( std::vector<uint32_t> &A,  std::vector<uint32_t> &B, std::vector<uint32_t> &C, int &num)
	//  C.resize(num);
	int n = 0;
	for (auto &a : A)
	{
		for (auto &b : B)
		{
			if (a == b)
			{
				C[n++] = a;
				if (n == num)
					break;
			}
		}
		if (n == num)
			break;
	}
}

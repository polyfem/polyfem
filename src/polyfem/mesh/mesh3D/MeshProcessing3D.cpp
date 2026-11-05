#include "MeshProcessing3D.hpp"
#include <polyfem/utils/Logger.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <queue>
#include <iterator>
#include <cassert>
#include <limits>

using namespace polyfem::mesh;
using namespace polyfem;
using namespace std;
using namespace Eigen;

namespace
{
	/// Fills rows (n_rows of them) with the (row, value) pairs passed to the
	/// callback by visit(callback), keeping the order in which they come.
	/// visit is called twice: once to count, once to fill.
	template <typename T, typename Visit>
	void fill_rows(JaggedArray<T> &rows, const int n_rows, const Visit &visit)
	{
		std::vector<uint32_t> counts(n_rows, 0);
		visit([&](const int r, const T &) { ++counts[r]; });
		rows.assign_sizes(counts);
		std::fill(counts.begin(), counts.end(), 0);
		visit([&](const int r, const T &value) { rows[r][counts[r]++] = value; });
	}

	/// A mesh under construction by the refinement: vertex positions, faces and
	/// cells; the connectivity is built afterwards.
	struct RefinedMesh
	{
		struct Cell
		{
			std::vector<uint32_t> vs;
			std::vector<uint32_t> fs;
			std::vector<uint8_t> fs_flag;
			bool hex = false;
			std::array<double, 3> kernel;
		};

		MeshType type;
		std::vector<std::array<double, 3>> points;
		std::vector<std::vector<uint32_t>> faces;
		std::vector<Cell> cells;

		Mesh3DStorage to_storage() const
		{
			Mesh3DStorage M;
			M.type = type;
			M.points.resize(3, points.size());
			for (size_t v = 0; v < points.size(); ++v)
				for (int j = 0; j < 3; j++)
					M.points(j, v) = points[v][j];
			for (const auto &f : faces)
				M.face_vs.push_back(f);
			for (const auto &c : cells)
			{
				M.cell_vs.push_back(c.vs);
				M.cell_input_vs.push_back_filled(0, 0);
				M.cell_fs.push_back(c.fs);
				M.cell_fs_flag.push_back(c.fs_flag);
				M.cell_hex.push_back(c.hex);
				M.cell_kernel.push_back(c.kernel);
			}
			return M;
		}
	};

	/// Scaled Jacobian quality of hexes given by their 8 vertices.
	bool hex_scaled_jacobian(const Eigen::MatrixXd &points, const std::vector<std::vector<uint32_t>> &hexes, Mesh_Quality &mq)
	{
		using namespace MeshProcessing3D;
		mq.ave_Jacobian = 0;
		mq.min_Jacobian = 1;
		mq.deviation_Jacobian = 0;
		mq.V_Js.resize(hexes.size() * 8);
		mq.V_Js.setZero();
		mq.H_Js.resize(hexes.size());
		mq.H_Js.setZero();

		for (uint32_t i = 0; i < hexes.size(); i++)
		{
			double hex_minJ = 1;
			for (uint32_t j = 0; j < 8; j++)
			{
				uint32_t v0, v1, v2, v3;
				v0 = hex_tetra_table[j][0];
				v1 = hex_tetra_table[j][1];
				v2 = hex_tetra_table[j][2];
				v3 = hex_tetra_table[j][3];

				Vector3d c0 = points.col(hexes[i][v0]);
				Vector3d c1 = points.col(hexes[i][v1]);
				Vector3d c2 = points.col(hexes[i][v2]);
				Vector3d c3 = points.col(hexes[i][v3]);

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
		mq.ave_Jacobian /= hexes.size();
		for (int i = 0; i < mq.H_Js.size(); i++)
			mq.deviation_Jacobian += (mq.H_Js[i] - mq.ave_Jacobian) * (mq.H_Js[i] - mq.ave_Jacobian);
		mq.deviation_Jacobian /= hexes.size();

		return true;
	}

	/// Builds the edges of the faces: edge_vs (sorted by (min, max)) and
	/// face_es. With mark_boundary, the edges of a single face are boundary;
	/// otherwise every edge starts as interior.
	void build_edges(Mesh3DStorage &hmi, const bool mark_boundary)
	{
		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> temp;
		temp.reserve(hmi.face_vs.n_entries());
		std::vector<uint32_t> face_sizes(hmi.n_faces());
		for (uint32_t i = 0; i < hmi.n_faces(); ++i)
		{
			const auto fvs = hmi.face_vs[i];
			const int vn = fvs.size();
			for (uint32_t j = 0; j < vn; ++j)
			{
				uint32_t v0 = fvs[j], v1 = fvs[(j + 1) % vn];
				if (v0 > v1)
					std::swap(v0, v1);
				temp.push_back(std::make_tuple(v0, v1, i, j));
			}
			face_sizes[i] = vn;
		}
		hmi.face_es.assign_sizes(face_sizes);
		std::sort(temp.begin(), temp.end());

		hmi.edge_vs.clear();
		hmi.edge_vs.reserve(temp.size() / 2);
		hmi.edge_boundary.clear();
		hmi.edge_boundary.reserve(temp.size() / 2);
		uint32_t E_num = 0;
		for (uint32_t i = 0; i < temp.size(); ++i)
		{
			if (i == 0 || (i != 0 && (std::get<0>(temp[i]) != std::get<0>(temp[i - 1]) || std::get<1>(temp[i]) != std::get<1>(temp[i - 1]))))
			{
				E_num++;
				hmi.edge_vs.push_back({{std::get<0>(temp[i]), std::get<1>(temp[i])}});
				hmi.edge_boundary.push_back(mark_boundary);
			}
			else if (mark_boundary)
				hmi.edge_boundary[E_num - 1] = false;

			hmi.face_es[std::get<2>(temp[i])][std::get<3>(temp[i])] = E_num - 1;
		}
	}
} // namespace

void MeshProcessing3D::build_connectivity(Mesh3DStorage &hmi)
{
	const int n_vertices = hmi.n_vertices();
	if (hmi.type == MeshType::TRI || hmi.type == MeshType::QUA || hmi.type == MeshType::H_SUR)
	{
		build_edges(hmi, true);
		// boundary
		hmi.vertex_boundary.assign(n_vertices, false);
		for (uint32_t i = 0; i < hmi.n_edges(); ++i)
			if (hmi.edge_boundary[i])
				hmi.vertex_boundary[hmi.edge_vs[i][0]] = hmi.vertex_boundary[hmi.edge_vs[i][1]] = true;
	}
	else if (hmi.type == MeshType::HEX)
	{
		// Faces from the hexes (cell_vs in navigation order).
		const int n_cells = hmi.cell_vs.size();
		std::vector<std::array<uint32_t, 4>> total_fs(n_cells * 6);
		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> tempF(n_cells * 6);
		std::array<uint32_t, 4> vs;
		for (uint32_t i = 0; i < n_cells; ++i)
		{
			const auto hvs = hmi.cell_vs[i];
			for (short j = 0; j < 6; j++)
			{
				for (short k = 0; k < 4; k++)
					vs[k] = hvs[hex_face_table[j][k]];
				uint32_t id = 6 * i + j;
				total_fs[id] = vs;
				std::sort(vs.begin(), vs.end());
				tempF[id] = std::make_tuple(vs[0], vs[1], vs[2], vs[3], id, i, j);
			}
		}
		hmi.cell_fs.assign_sizes(std::vector<uint32_t>(n_cells, 6));
		std::sort(tempF.begin(), tempF.end());
		hmi.face_vs.clear();
		hmi.face_vs.reserve(tempF.size() / 3, 4 * (tempF.size() / 3));
		hmi.face_boundary.clear();
		uint32_t F_num = 0;
		for (uint32_t i = 0; i < tempF.size(); ++i)
		{
			if (i == 0 || (i != 0 && (std::get<0>(tempF[i]) != std::get<0>(tempF[i - 1]) || std::get<1>(tempF[i]) != std::get<1>(tempF[i - 1]) || std::get<2>(tempF[i]) != std::get<2>(tempF[i - 1]) || std::get<3>(tempF[i]) != std::get<3>(tempF[i - 1]))))
			{
				F_num++;
				hmi.face_vs.push_back(total_fs[std::get<4>(tempF[i])]);
				hmi.face_boundary.push_back(true);
			}
			else
				hmi.face_boundary[F_num - 1] = false;

			hmi.cell_fs[std::get<5>(tempF[i])][std::get<6>(tempF[i])] = F_num - 1;
		}

		build_edges(hmi, false);
		// boundary
		hmi.vertex_boundary.assign(n_vertices, false);
		for (uint32_t i = 0; i < hmi.n_faces(); ++i)
			if (hmi.face_boundary[i])
				for (uint32_t j = 0; j < 4; ++j)
				{
					uint32_t eid = hmi.face_es[i][j];
					hmi.edge_boundary[eid] = true;
					hmi.vertex_boundary[hmi.edge_vs[eid][0]] = hmi.vertex_boundary[hmi.edge_vs[eid][1]] = true;
				}
	}
	else if (hmi.type == MeshType::HYB || hmi.type == MeshType::TET)
	{
		// A face is on the boundary if it belongs to a single cell.
		hmi.face_boundary.assign(hmi.n_faces(), false);
		for (int c = 0; c < hmi.n_cells(); ++c)
			for (const auto f : hmi.cell_fs[c])
				hmi.face_boundary[f] = !hmi.face_boundary[f];

		build_edges(hmi, false);
		// boundary
		hmi.vertex_boundary.assign(n_vertices, false);
		for (uint32_t i = 0; i < hmi.n_faces(); ++i)
			if (hmi.face_boundary[i])
				for (uint32_t j = 0; j < hmi.face_vs.row_size(i); ++j)
				{
					uint32_t eid = hmi.face_es[i][j];
					hmi.edge_boundary[eid] = true;
					hmi.vertex_boundary[hmi.face_vs[i][j]] = true;
				}
	}

	const int n_edges = hmi.n_edges(), n_faces = hmi.n_faces(), n_cells = hmi.n_cells();

	// f_nhs
	fill_rows(hmi.face_hs, n_faces, [&](const auto &emit) {
		for (uint32_t i = 0; i < n_cells; i++)
			for (const auto f : hmi.cell_fs[i])
				emit(f, i);
	});
	// e_nfs, v_nfs
	fill_rows(hmi.edge_fs, n_edges, [&](const auto &emit) {
		for (uint32_t i = 0; i < n_faces; i++)
			for (const auto e : hmi.face_es[i])
				emit(e, i);
	});
	fill_rows(hmi.vertex_fs, n_vertices, [&](const auto &emit) {
		for (uint32_t i = 0; i < n_faces; i++)
			for (const auto v : hmi.face_vs[i])
				emit(v, i);
	});
	// v_nes, v_nvs
	fill_rows(hmi.vertex_es, n_vertices, [&](const auto &emit) {
		for (uint32_t i = 0; i < n_edges; i++)
		{
			emit(hmi.edge_vs[i][0], i);
			emit(hmi.edge_vs[i][1], i);
		}
	});
	fill_rows(hmi.vertex_vs, n_vertices, [&](const auto &emit) {
		for (uint32_t i = 0; i < n_edges; i++)
		{
			emit(hmi.edge_vs[i][0], hmi.edge_vs[i][1]);
			emit(hmi.edge_vs[i][1], hmi.edge_vs[i][0]);
		}
	});
	// e_nhs
	hmi.edge_hs.clear();
	hmi.edge_hs.reserve(n_edges, 6 * n_edges);
	std::vector<uint32_t> nhs;
	for (uint32_t i = 0; i < n_edges; i++)
	{
		nhs.clear();
		for (const auto nfid : hmi.edge_fs[i])
			nhs.insert(nhs.end(), hmi.face_hs[nfid].begin(), hmi.face_hs[nfid].end());
		std::sort(nhs.begin(), nhs.end());
		nhs.erase(std::unique(nhs.begin(), nhs.end()), nhs.end());
		hmi.edge_hs.push_back(nhs);
	}
	fill_rows(hmi.cell_es, n_cells, [&](const auto &emit) {
		for (uint32_t i = 0; i < n_edges; i++)
			for (const auto c : hmi.edge_hs[i])
				emit(c, i);
	});

	// Local index of each face in its cells (aligned with face_hs).
	const auto build_face_slots = [&]() {
		fill_rows(hmi.face_hs_slot, n_faces, [&](const auto &emit) {
			for (uint32_t i = 0; i < n_cells; i++)
				for (uint32_t j = 0; j < hmi.cell_fs.row_size(i); j++)
					emit(hmi.cell_fs[i][j], uint8_t(j));
		});
	};

	// v_nhs; ordering fs for hex
	if (hmi.type != MeshType::HYB && hmi.type != MeshType::TET)
	{
		build_face_slots();
		return;
	}

	JaggedArray<uint32_t> cell_vs;
	cell_vs.reserve(n_cells, hmi.cell_input_vs.n_entries());
	vector<uint32_t> vs, new_vs;
	for (uint32_t i = 0; i < n_cells; i++)
	{
		vs.clear();
		for (const auto fid : hmi.cell_fs[i])
			vs.insert(vs.end(), hmi.face_vs[fid].begin(), hmi.face_vs[fid].end());
		sort(vs.begin(), vs.end());
		vs.erase(unique(vs.begin(), vs.end()), vs.end());

		bool degree3 = true;
		for (auto vid : vs)
		{
			int nv = 0;
			for (auto nvid : hmi.vertex_vs[vid])
				if (find(vs.begin(), vs.end(), nvid) != vs.end())
					nv++;
			if (nv != 3)
			{
				degree3 = false;
				break;
			}
		}

		if (hmi.cell_hex[i] && (vs.size() != 8 || !degree3))
			hmi.cell_hex[i] = false;

		if (hmi.cell_hex[i])
		{
			const auto hfs = hmi.cell_fs[i];
			const auto hfs_flag = hmi.cell_fs_flag[i];
			int top_fid = hfs[0];
			new_vs = hmi.face_vs[top_fid].to_vector();

			std::set<uint32_t> s_model(vs.begin(), vs.end());
			std::set<uint32_t> s_pattern(hmi.face_vs[top_fid].begin(), hmi.face_vs[top_fid].end());
			vector<uint32_t> vs_left;
			std::set_difference(s_model.begin(), s_model.end(), s_pattern.begin(), s_pattern.end(), std::back_inserter(vs_left));

			for (auto vid : hmi.face_vs[top_fid])
				for (auto nvid : hmi.vertex_vs[vid])
					if (find(vs_left.begin(), vs_left.end(), nvid) != vs_left.end())
					{
						new_vs.push_back(nvid);
						break;
					}

			function<int(vector<uint32_t> &, int &)> WHICH_F = [&](vector<uint32_t> &vs0, int &f_flag) -> int {
				int which_f = -1;
				sort(vs0.begin(), vs0.end());
				for (uint32_t j = 0; j < hfs.size(); j++)
				{
					auto fid = hfs[j];
					vector<uint32_t> vs1 = hmi.face_vs[fid].to_vector();
					sort(vs1.begin(), vs1.end());
					if (vs0.size() == vs1.size() && std::equal(vs0.begin(), vs0.end(), vs1.begin()))
					{
						f_flag = hfs_flag[j];
						which_f = fid;
						break;
					}
				}
				return which_f;
			};

			vector<uint32_t> fs;
			vector<uint8_t> fs_flag;
			fs_flag.push_back(hfs_flag[0]);
			fs.push_back(top_fid);
			vector<uint32_t> vs_temp;

			vs_temp.insert(vs_temp.end(), new_vs.begin() + 4, new_vs.end());
			int f_flag = -1;
			int bottom_fid = WHICH_F(vs_temp, f_flag);
			fs_flag.push_back(f_flag);
			fs.push_back(bottom_fid);

			vs_temp = {new_vs[0], new_vs[1], new_vs[4], new_vs[5]};
			f_flag = -1;
			int front_fid = WHICH_F(vs_temp, f_flag);
			fs_flag.push_back(f_flag);
			fs.push_back(front_fid);

			vs_temp = {new_vs[2], new_vs[3], new_vs[6], new_vs[7]};
			f_flag = -1;
			int back_fid = WHICH_F(vs_temp, f_flag);
			fs_flag.push_back(f_flag);
			fs.push_back(back_fid);

			vs_temp = {new_vs[1], new_vs[2], new_vs[5], new_vs[6]};
			f_flag = -1;
			int left_fid = WHICH_F(vs_temp, f_flag);
			fs_flag.push_back(f_flag);
			fs.push_back(left_fid);

			vs_temp = {new_vs[3], new_vs[0], new_vs[7], new_vs[4]};
			f_flag = -1;
			int right_fid = WHICH_F(vs_temp, f_flag);
			fs_flag.push_back(f_flag);
			fs.push_back(right_fid);

			assert(fs.size() == hfs.size());
			std::copy(fs.begin(), fs.end(), hfs.begin());
			std::copy(fs_flag.begin(), fs_flag.end(), hfs_flag.begin());
			cell_vs.push_back(new_vs);
		}
		else
			cell_vs.push_back(vs);
	}
	hmi.cell_vs = std::move(cell_vs);
	fill_rows(hmi.vertex_hs, n_vertices, [&](const auto &emit) {
		for (uint32_t i = 0; i < n_cells; i++)
			for (const auto v : hmi.cell_vs[i])
				emit(v, i);
	});
	build_face_slots();
}

void MeshProcessing3D::reorder_hex_mesh_propogation(Mesh3DStorage &hmi)
{
	// connected components
	vector<bool> H_tag(hmi.n_cells(), false);
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
						Fvs[i].push_back(hmi.cell_vs[hid][hex_face_table[i][j]]);
				fvs_sorted = Fvs;
				for (auto &vs : fvs_sorted)
					sort(vs.begin(), vs.end());

				for (auto fid : hmi.cell_fs[hid])
					if (!hmi.face_boundary[fid])
					{
						int nhid = hmi.face_hs[fid][0];
						if (nhid == hid)
							nhid = hmi.face_hs[fid][1];

						if (!H_tag[nhid])
						{
							pool.push_back(nhid);
							H_tag[nhid] = true;

							vector<uint32_t> fvs = hmi.face_vs[fid].to_vector();
							sort(fvs.begin(), fvs.end());

							int f_ind = -1;
							for (uint32_t i = 0; i < 6; i++)
								if (std::equal(fvs.begin(), fvs.end(), fvs_sorted[i].begin()))
								{
									f_ind = i;
									break;
								}
							vector<uint32_t> hvs = hmi.cell_vs[nhid].to_vector();
							vector<uint32_t> topvs = Fvs[f_ind];
							std::reverse(topvs.begin(), topvs.end());
							vector<uint32_t> bottomvs;
							for (uint32_t i = 0; i < 4; i++)
							{
								for (auto nvid : hmi.vertex_vs[topvs[i]])
									if (nvid != topvs[(i + 3) % 4] && nvid != topvs[(i + 1) % 4]
										&& std::find(hvs.begin(), hvs.end(), nvid) != hvs.end())
									{
										bottomvs.push_back(nvid);
										break;
									}
							}
							const auto new_vs = hmi.cell_vs[nhid];
							assert(topvs.size() + bottomvs.size() == new_vs.size());
							std::copy(bottomvs.begin(), bottomvs.end(), std::copy(topvs.begin(), topvs.end(), new_vs.begin()));
						}
					}
			}
			if (pool.size())
			{
				group_ = pool;
				group.insert(group.end(), pool.begin(), pool.end());
			}
			else
				break;
		}
		Groups.push_back(group);
	}
	// direction
	for (const auto &group : Groups)
	{
		Mesh_Quality mq1, mq2;
		std::vector<std::vector<uint32_t>> hexes;
		hexes.reserve(group.size());
		for (auto hid : group)
			hexes.push_back(hmi.cell_vs[hid].to_vector());
		hex_scaled_jacobian(hmi.points, hexes, mq1);
		if (mq1.min_Jacobian > 0)
			continue;
		for (auto &h : hexes)
		{
			swap(h[1], h[3]);
			swap(h[5], h[7]);
		}
		hex_scaled_jacobian(hmi.points, hexes, mq2);
		if (mq2.ave_Jacobian > mq1.ave_Jacobian)
		{
			for (size_t k = 0; k < group.size(); ++k)
				std::copy(hexes[k].begin(), hexes[k].end(), hmi.cell_vs[group[k]].begin());
		}
	}
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
	// Reorients the hexes (only) consistently: build a hex-only mesh, propagate
	// the orientation across shared faces, and copy the vertex orders back.
	if (std::none_of(hmi.cell_hex.begin(), hmi.cell_hex.end(), [](const uint8_t hex) { return hex; }))
		return;

	// reorder_hex_mesh_propogation only uses the connectivity rebuilt below
	// (vertex neighbors and face/element adjacency) and the points.
	Mesh3DStorage mesh;
	mesh.type = MeshType::HEX;
	mesh.points = hmi.points;
	vector<int> Ele_map_reverse;
	for (int c = 0; c < hmi.n_cells(); ++c)
	{
		if (!hmi.cell_hex[c])
			continue;
		mesh.cell_vs.push_back(hmi.cell_vs[c]);
		Ele_map_reverse.push_back(c);
	}

	build_connectivity(mesh);
	reorder_hex_mesh_propogation(mesh);

	for (int c = 0; c < mesh.cell_vs.size(); ++c)
	{
		const auto vs = mesh.cell_vs[c];
		std::copy(vs.begin(), vs.end(), hmi.cell_vs[Ele_map_reverse[c]].begin());
	}
}

void MeshProcessing3D::refine_catmul_clark_polar(Mesh3DStorage &M, int iter, bool reverse, std::vector<int> &Parents)
{

	for (int i = 0; i < iter; i++)
	{

		std::vector<int> Refinement_Levels;
		ele_subdivison_levels(M, Refinement_Levels);

		RefinedMesh M_;
		M_.type = MeshType::HYB;

		vector<int> E2V(M.n_edges()), F2V(M.n_faces()), Ele2V(M.n_cells());

		const auto add_vertex = [&](const std::array<double, 3> &p) {
			M_.points.push_back(p);
			return int(M_.points.size()) - 1;
		};

		for (int v = 0; v < M.n_vertices(); ++v)
			add_vertex({{M.points(0, v), M.points(1, v), M.points(2, v)}});

		for (int e = 0; e < M.n_edges(); ++e)
		{
			Vector3d center;
			center.setZero();
			for (auto vid : M.edge_vs[e])
				center += M.points.col(vid);
			center /= M.edge_vs[e].size();
			E2V[e] = add_vertex({{center[0], center[1], center[2]}});
		}
		for (int f = 0; f < M.n_faces(); ++f)
		{
			Vector3d center;
			center.setZero();
			for (auto vid : M.face_vs[f])
				center += M.points.col(vid);
			center /= M.face_vs[f].size();
			F2V[f] = add_vertex({{center[0], center[1], center[2]}});
		}
		for (int c = 0; c < M.n_cells(); ++c)
		{
			if (!M.cell_hex[c])
				continue;
			Ele2V[c] = add_vertex(M.cell_kernel[c]);
		}
		// new elements
		std::vector<std::array<uint32_t, 4>> total_fs;
		total_fs.reserve(M.n_cells() * 8 * 6);
		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> tempF;
		tempF.reserve(M.n_cells() * 8 * 6);
		std::array<uint32_t, 4> vs;

		int elen = 0, fn = 0;
		for (int c = 0; c < M.n_cells(); ++c)
		{
			const auto ele_vs_in = M.cell_vs[c];
			const auto ele_fs = M.cell_fs[c];
			if (M.cell_hex[c])
			{
				for (auto vid : ele_vs_in)
				{
					// top 4 vs
					vector<int> top_vs(4);
					top_vs[0] = vid;
					int fid = -1;
					for (auto nfid : M.vertex_fs[vid])
						if (find(ele_fs.begin(), ele_fs.end(), nfid) != ele_fs.end())
						{
							fid = nfid;
							break;
						}
					assert(fid != -1);
					top_vs[2] = F2V[fid];

					int v_ind = find(M.face_vs[fid].begin(), M.face_vs[fid].end(), vid) - M.face_vs[fid].begin();
					int e_pre = M.face_es[fid][(v_ind - 1 + 4) % 4];
					int e_aft = M.face_es[fid][v_ind];
					top_vs[1] = E2V[e_pre];
					top_vs[3] = E2V[e_aft];
					// bottom 4 vs
					vector<int> bottom_vs(4);

					const auto nvs = M.vertex_vs[vid];
					int e_per = -1;
					for (auto nvid : nvs)
						if (find(ele_vs_in.begin(), ele_vs_in.end(), nvid) != ele_vs_in.end())
						{
							if (nvid != M.edge_vs[e_pre][0] && nvid != M.edge_vs[e_pre][1] && nvid != M.edge_vs[e_aft][0] && nvid != M.edge_vs[e_aft][1])
							{
								vector<uint32_t> sharedes, es0 = M.vertex_es[vid].to_vector(), es1 = M.vertex_es[nvid].to_vector();
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
					bottom_vs[2] = Ele2V[c];

					int f_pre = -1;
					vector<uint32_t> sharedfs, fs0 = M.edge_fs[e_pre].to_vector(), fs1 = M.edge_fs[e_per].to_vector();
					sort(fs0.begin(), fs0.end());
					sort(fs1.begin(), fs1.end());
					set_intersection(fs0.begin(), fs0.end(), fs1.begin(), fs1.end(), back_inserter(sharedfs));
					for (auto sfid : sharedfs)
						if (find(ele_fs.begin(), ele_fs.end(), sfid) != ele_fs.end())
						{
							f_pre = sfid;
							break;
						}
					assert(f_pre != -1);

					int f_aft = -1;
					sharedfs.clear();
					fs0 = M.edge_fs[e_aft].to_vector();
					fs1 = M.edge_fs[e_per].to_vector();
					sort(fs0.begin(), fs0.end());
					sort(fs1.begin(), fs1.end());
					set_intersection(fs0.begin(), fs0.end(), fs1.begin(), fs1.end(), back_inserter(sharedfs));
					for (auto sfid : sharedfs)
						if (find(ele_fs.begin(), ele_fs.end(), sfid) != ele_fs.end())
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
					RefinedMesh::Cell ele_;
					elen++;
					ele_.fs.resize(6, -1);
					ele_.fs_flag.resize(6, 1);
					ele_.hex = true;

					Vector3d center;
					center.setZero();
					for (auto evid : ele_vs)
						for (int j = 0; j < 3; j++)
							center[j] += M_.points[evid][j];
					center /= ele_vs.size();
					for (int j = 0; j < 3; j++)
						ele_.kernel[j] = center[j];

					M_.cells.push_back(std::move(ele_));
					Parents.push_back(c);
				}
			}
			else
			{
				const std::array<double, 3> kernel = M.cell_kernel[c];
				int level = Refinement_Levels[c];
				if (reverse)
					level = 1;
				// local_V2V
				std::vector<std::vector<int>> local_V2Vs;
				std::map<int, int> local_vi_map;
				for (auto vid : ele_vs_in)
				{
					std::vector<int> v2v;
					v2v.push_back(vid);
					for (int r = 0; r < level; r++)
					{
						std::array<double, 3> p;
						for (int j = 0; j < 3; j++)
							p[j] = M_.points[vid][j] + (kernel[j] - M_.points[vid][j]) * (r + 1.0) / (double)(level + 1);
						if (reverse)
						{
							for (int j = 0; j < 3; j++)
								p[j] = M_.points[vid][j] + (M_.points[vid][j] - kernel[j]) * (r + 1.0) / (double)(level + 1);
						}
						v2v.push_back(add_vertex(p));
					}
					local_vi_map[vid] = local_V2Vs.size();
					local_V2Vs.push_back(v2v);
				}
				// local_E2V
				vector<uint32_t> es;
				for (auto fid : ele_fs)
					es.insert(es.end(), M.face_es[fid].begin(), M.face_es[fid].end());
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
						Vector3d center;
						center.setZero();
						for (auto vid : M.edge_vs[eid])
							for (int j = 0; j < 3; j++)
								center[j] += M_.points[local_V2Vs[local_vi_map[vid]][r + 1]][j];
						center /= M.edge_vs[eid].size();
						e2v.push_back(add_vertex({{center[0], center[1], center[2]}}));
					}
					local_ei_map[eid] = local_E2Vs.size();
					local_E2Vs.push_back(e2v);
				}
				// local_F2V
				std::vector<std::vector<int>> local_F2Vs;
				std::map<int, int> local_fi_map;
				for (auto fid : ele_fs)
				{
					std::vector<int> f2v;
					f2v.push_back(F2V[fid]);
					for (int r = 0; r < level; r++)
					{
						Vector3d center;
						center.setZero();
						for (auto vid : M.face_vs[fid])
							for (int j = 0; j < 3; j++)
								center[j] += M_.points[local_V2Vs[local_vi_map[vid]][r + 1]][j];
						center /= M.face_vs[fid].size();
						f2v.push_back(add_vertex({{center[0], center[1], center[2]}}));
					}
					local_fi_map[fid] = local_F2Vs.size();
					local_F2Vs.push_back(f2v);
				}
				// polyhedron fs
				int local_fn = 0;
				for (auto fid : ele_fs)
				{
					const auto fvs = M.face_vs[fid];
					const auto fes = M.face_es[fid];
					int fvn = fvs.size();
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
				RefinedMesh::Cell ele_;
				elen++;
				ele_.fs.resize(local_fn, -1);
				ele_.fs_flag.resize(local_fn, 1);
				ele_.hex = false;
				ele_.kernel = kernel;
				M_.cells.push_back(std::move(ele_));
				Parents.push_back(c);
				// hex
				for (int r = 0; r < level; r++)
				{
					for (auto fid : ele_fs)
					{
						const auto fvs = M.face_vs[fid];
						const auto fes = M.face_es[fid];
						int fvn = fvs.size();
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
							RefinedMesh::Cell hex;
							elen++;
							hex.fs.resize(6, -1);
							hex.fs_flag.resize(6, 1);
							hex.hex = true;

							Vector3d center;
							center.setZero();
							for (auto vid : ele_vs)
								for (int j = 0; j < 3; j++)
									center[j] += M_.points[vid][j];
							center /= ele_vs.size();
							for (int j = 0; j < 3; j++)
								hex.kernel[j] = center[j];
							M_.cells.push_back(std::move(hex));
							Parents.push_back(c);
						}
					}
				}
			}
		}
		// Fs
		std::sort(tempF.begin(), tempF.end());
		M_.faces.reserve(tempF.size() / 3);
		uint32_t F_num = 0;
		for (uint32_t i = 0; i < tempF.size(); ++i)
		{
			if (i == 0 || (i != 0 && (std::get<0>(tempF[i]) != std::get<0>(tempF[i - 1]) || std::get<1>(tempF[i]) != std::get<1>(tempF[i - 1]) || std::get<2>(tempF[i]) != std::get<2>(tempF[i - 1]) || std::get<3>(tempF[i]) != std::get<3>(tempF[i - 1]))))
			{
				F_num++;
				const auto &fvs = total_fs[std::get<4>(tempF[i])];
				M_.faces.emplace_back(fvs.begin(), fvs.end());
			}

			M_.cells[std::get<5>(tempF[i])].fs[std::get<6>(tempF[i])] = F_num - 1;
		}

		Mesh3DStorage refined = M_.to_storage();
		build_connectivity(refined);
		orient_volume_mesh(refined);
		build_connectivity(refined);

		M = std::move(refined);
	}
}

void MeshProcessing3D::refine_red_refinement_tet(Mesh3DStorage &M, int iter)
{
	for (int i = 0; i < iter; i++)
	{
		RefinedMesh M_;
		M_.type = MeshType::TET;

		vector<int> E2V(M.n_edges());

		M_.points.reserve(M.n_vertices() + M.n_edges());
		for (int v = 0; v < M.n_vertices(); ++v)
			M_.points.push_back({{M.points(0, v), M.points(1, v), M.points(2, v)}});

		for (int e = 0; e < M.n_edges(); ++e)
		{
			Vector3d center;
			center.setZero();
			for (auto vid : M.edge_vs[e])
				center += M.points.col(vid);
			center /= M.edge_vs[e].size();
			M_.points.push_back({{center[0], center[1], center[2]}});
			E2V[e] = M_.points.size() - 1;
		}

		// The edge shared by two vertices (the edges around a vertex are sorted).
		auto shared_edge = [&](int v0, int v1, int &e) -> bool {
			const auto es0 = M.vertex_es[v0], es1 = M.vertex_es[v1];
			auto it0 = es0.begin(), it1 = es1.begin();
			while (it0 != es0.end() && it1 != es1.end())
			{
				if (*it0 < *it1)
					++it0;
				else if (*it1 < *it0)
					++it1;
				else
				{
					e = *it0;
					return true;
				}
			}
			return false;
		};

		const auto kernel_of = [&](const std::vector<uint32_t> &vs) {
			Vector3d center;
			center.setZero();
			for (const auto &evid : vs)
				for (int j = 0; j < 3; j++)
					center[j] += M_.points[evid][j];
			center /= vs.size();
			return std::array<double, 3>{{center[0], center[1], center[2]}};
		};

		std::vector<bool> e_flag(M.n_edges(), false);

		M_.cells.reserve(8 * M.n_cells());
		for (int c = 0; c < M.n_cells(); ++c)
		{ // 1 --> 8
			const auto ele_vs = M.cell_vs[c];
			const auto ele_fs = M.cell_fs[c];

			for (short i = 0; i < 4; i++)
			{ // four corners
				RefinedMesh::Cell ele_;
				ele_.fs.resize(4, -1);
				ele_.fs_flag.resize(4, 1);
				ele_.hex = false;

				ele_.vs.push_back(ele_vs[i]);
				for (short j = 0; j < 4; j++)
				{
					if (j == i)
						continue;
					int v0 = ele_vs[i], v1 = ele_vs[j];
					int e = -1;
					if (shared_edge(v0, v1, e))
						ele_.vs.push_back(E2V[e]);
				}
				ele_.kernel = kernel_of(ele_.vs);
				M_.cells.push_back(std::move(ele_));
			}

			// 6 edges
			std::vector<int> edges(6);
			for (int i = 0; i < 6; i++)
			{
				int v0 = ele_vs[tet_edges[i][0]], v1 = ele_vs[tet_edges[i][1]];
				int e = -1;
				if (shared_edge(v0, v1, e))
					edges[i] = e;
			}
			// Split the inner octahedron along its shortest diagonal, i.e. between the midpoints
			// of two opposite edges (k, 5 - k). A fixed choice of diagonal makes the tets
			// degenerate under repeated refinement, since build_connectivity sorts ele.vs.
			int diag = 0;
			double min_length = std::numeric_limits<double>::max();
			for (int k = 0; k < 3; ++k)
			{
				const double length = (Map<const Vector3d>(M_.points[E2V[edges[k]]].data()) - Map<const Vector3d>(M_.points[E2V[edges[5 - k]]].data())).squaredNorm();
				if (length < min_length)
				{
					min_length = length;
					diag = k;
				}
			}
			int lv0 = E2V[edges[diag]], lv1 = E2V[edges[5 - diag]];
			e_flag[edges[diag]] = true;
			e_flag[edges[5 - diag]] = true;
			for (short i = 0; i < 4; i++)
			{ // four faces
				RefinedMesh::Cell ele_;
				ele_.fs.resize(4, -1);
				ele_.fs_flag.resize(4, 1);
				ele_.hex = false;

				ele_.vs.push_back(lv0);
				ele_.vs.push_back(lv1);
				for (short j = 0; j < 3; j++)
				{
					int c_e = M.face_es[ele_fs[i]][j];
					if (e_flag[c_e])
						continue;
					ele_.vs.push_back(E2V[c_e]);
				}
				ele_.kernel = kernel_of(ele_.vs);
				M_.cells.push_back(std::move(ele_));
			}
			e_flag[edges[diag]] = e_flag[edges[5 - diag]] = false;
		}

		const auto point = [&](const uint32_t v) { return Vector3d(M_.points[v][0], M_.points[v][1], M_.points[v][2]); };

		// orient tets
		const auto t = M.cell_vs[0];
		Vector3d c0 = M.points.col(t[0]);
		Vector3d c1 = M.points.col(t[1]);
		Vector3d c2 = M.points.col(t[2]);
		Vector3d c3 = M.points.col(t[3]);
		bool signed_volume = a_jacobian(c0, c1, c2, c3) > 0 ? true : false;

		for (auto &ele : M_.cells)
		{
			c0 = point(ele.vs[0]);
			c1 = point(ele.vs[1]);
			c2 = point(ele.vs[2]);
			c3 = point(ele.vs[3]);
			bool sign = a_jacobian(c0, c1, c2, c3) > 0 ? true : false;
			if (sign != signed_volume)
				std::swap(ele.vs[1], ele.vs[3]);
		}

		// Fs
		std::vector<std::array<uint32_t, 3>> total_fs;
		total_fs.reserve(M_.cells.size() * 4);
		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> tempF;
		tempF.reserve(M_.cells.size() * 4);
		std::array<uint32_t, 3> vs;
		for (uint32_t c = 0; c < M_.cells.size(); ++c)
		{
			const auto &ele = M_.cells[c];
			for (int i = 0; i < 4; i++)
			{
				vs[0] = ele.vs[tet_faces[i][0]];
				vs[1] = ele.vs[tet_faces[i][1]];
				vs[2] = ele.vs[tet_faces[i][2]];
				total_fs.push_back(vs);
				std::sort(vs.begin(), vs.end());
				tempF.push_back(std::make_tuple(vs[0], vs[1], vs[2], c * 4 + i, c, i));
			}
		}
		std::sort(tempF.begin(), tempF.end());
		M_.faces.reserve(tempF.size() / 3);
		uint32_t F_num = 0;
		for (uint32_t i = 0; i < tempF.size(); ++i)
		{
			if (i == 0 || (i != 0 && (std::get<0>(tempF[i]) != std::get<0>(tempF[i - 1]) || std::get<1>(tempF[i]) != std::get<1>(tempF[i - 1]) || std::get<2>(tempF[i]) != std::get<2>(tempF[i - 1]))))
			{
				F_num++;
				const auto &fvs = total_fs[std::get<3>(tempF[i])];
				M_.faces.emplace_back(fvs.begin(), fvs.end());
			}

			M_.cells[std::get<4>(tempF[i])].fs[std::get<5>(tempF[i])] = F_num - 1;
		}

		Mesh3DStorage refined = M_.to_storage();
		build_connectivity(refined);
		orient_volume_mesh(refined);
		build_connectivity(refined);

		M = std::move(refined);
	}
}

void MeshProcessing3D::orient_surface_mesh(Mesh3DStorage &hmi)
{

	vector<bool> flag(hmi.n_faces(), true);
	flag[0] = false;

	std::queue<uint32_t> pf_temp;
	pf_temp.push(0);
	while (!pf_temp.empty())
	{
		uint32_t fid = pf_temp.front();
		pf_temp.pop();
		const auto fvs = hmi.face_vs[fid];
		for (auto eid : hmi.face_es[fid])
			for (auto nfid : hmi.edge_fs[eid])
			{
				if (!flag[nfid])
					continue;
				uint32_t v0 = hmi.edge_vs[eid][0], v1 = hmi.edge_vs[eid][1];
				int32_t v0_pos = std::find(fvs.begin(), fvs.end(), v0) - fvs.begin();
				int32_t v1_pos = std::find(fvs.begin(), fvs.end(), v1) - fvs.begin();

				if ((v0_pos + 1) % fvs.size() != v1_pos)
					swap(v0, v1);

				const auto nfvs = hmi.face_vs[nfid];
				int32_t v0_pos_ = std::find(nfvs.begin(), nfvs.end(), v0) - nfvs.begin();
				int32_t v1_pos_ = std::find(nfvs.begin(), nfvs.end(), v1) - nfvs.begin();

				if ((v0_pos_ + 1) % nfvs.size() == v1_pos_)
					std::reverse(nfvs.begin(), nfvs.end());

				pf_temp.push(nfid);
				flag[nfid] = false;
			}
	}
	double res = 0;
	Vector3d ori;
	ori.setZero();
	for (int f = 0; f < hmi.n_faces(); ++f)
	{
		const auto fvs = hmi.face_vs[f];
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
		for (uint32_t i = 0; i < hmi.n_faces(); i++)
		{
			const auto fvs = hmi.face_vs[i];
			std::reverse(fvs.begin(), fvs.end());
		}
	}
}

void MeshProcessing3D::orient_volume_mesh(Mesh3DStorage &hmi)
{
	// surface orienting
	Mesh3DStorage M_sur;
	M_sur.type = MeshType::H_SUR;
	int bvn = 0;
	for (int v = 0; v < hmi.n_vertices(); ++v)
		if (hmi.vertex_boundary[v])
			bvn++;
	M_sur.points.resize(3, bvn);
	bvn = 0;
	vector<int> V_map(hmi.n_vertices(), -1), V_map_reverse;
	for (int v = 0; v < hmi.n_vertices(); ++v)
		if (hmi.vertex_boundary[v])
		{
			M_sur.points.col(bvn++) = hmi.points.col(v);
			V_map[v] = V_map_reverse.size();
			V_map_reverse.push_back(v);
		}
	std::vector<uint32_t> boundary_fvs;
	for (int f = 0; f < hmi.n_faces(); ++f)
		if (hmi.face_boundary[f])
		{
			boundary_fvs.clear();
			for (const auto v : hmi.face_vs[f])
				boundary_fvs.push_back(V_map[v]);
			M_sur.face_vs.push_back(boundary_fvs);
		}
	build_connectivity(M_sur);
	orient_surface_mesh(M_sur);

	int fn_ = 0;
	for (int f = 0; f < hmi.n_faces(); ++f)
		if (hmi.face_boundary[f])
		{
			const auto fvs = hmi.face_vs[f];
			for (int j = 0; j < fvs.size(); j++)
				fvs[j] = V_map_reverse[M_sur.face_vs[fn_][j]];
			fn_++;
		}
	// volume orienting
	vector<bool> F_tag(hmi.n_faces(), true);
	std::vector<short> F_visit(hmi.n_faces(), 0); // 0 un-visited, 1 visited once, 2 visited twice
	for (uint32_t j = 0; j < hmi.n_faces(); j++)
		if (hmi.face_boundary[j])
		{
			F_visit[j]++;
		}
	std::vector<bool> F_state(hmi.n_faces(), false); // false is the reverse direction, true is the same direction
	std::vector<bool> P_visit(hmi.n_cells(), false);
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
			uint32_t pid = hmi.face_hs[ca][0];
			if (P_visit[pid])
				if (hmi.face_hs.row_size(ca) == 2)
					pid = hmi.face_hs[ca][1];
			if (P_visit[pid])
			{
				logger().error("bug");
			}
			const auto fs = hmi.cell_fs[pid];
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
				const auto fvs = hmi.face_vs[fid];
				for (auto eid : hmi.face_es[fid])
					for (auto nfid : hmi.edge_fs[eid])
					{

						if (F_tag[nfid])
							continue;
						uint32_t v0 = hmi.edge_vs[eid][0], v1 = hmi.edge_vs[eid][1];
						int32_t v0_pos = std::find(fvs.begin(), fvs.end(), v0) - fvs.begin();
						int32_t v1_pos = std::find(fvs.begin(), fvs.end(), v1) - fvs.begin();

						if ((v0_pos + 1) % fvs.size() != v1_pos)
							std::swap(v0, v1);

						const auto nfvs = hmi.face_vs[nfid];
						int32_t v0_pos_ = std::find(nfvs.begin(), nfvs.end(), v0) - nfvs.begin();
						int32_t v1_pos_ = std::find(nfvs.begin(), nfvs.end(), v1) - nfvs.begin();

						if (F_state[fid])
						{
							if ((v0_pos_ + 1) % nfvs.size() == v1_pos_)
								F_state[nfid] = false;
							else
								F_state[nfid] = true;
						}
						else if (!F_state[fid])
						{
							if ((v0_pos_ + 1) % nfvs.size() == v1_pos_)
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
			const auto fs_flag = hmi.cell_fs_flag[pid];
			for (uint32_t j = 0; j < fs.size(); j++)
				fs_flag[j] = F_state[fs[j]];
		}
	}
}

void MeshProcessing3D::ele_subdivison_levels(const Mesh3DStorage &hmi, std::vector<int> &Ls)
{

	Ls.clear();
	Ls.resize(hmi.n_cells(), 1);
	std::vector<double> volumes(hmi.n_cells(), 0);

	auto compute_volume = [&](const int id, double &vol) {
		Vector3d ori;
		ori.setZero();
		for (auto f : hmi.cell_fs[id])
		{
			const auto fvs = hmi.face_vs[f];
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

	for (int c = 0; c < hmi.n_cells(); ++c)
		compute_volume(c, volumes[c]);

	double ave_volume = 0;
	for (const auto &v : volumes)
		ave_volume += v;
	ave_volume /= volumes.size();
	for (int i = 0; i < Ls.size(); i++)
		if (!hmi.cell_hex[i])
		{
			Ls[i] = volumes[i] / ave_volume;
			if (Ls[i] < 1)
				Ls[i] = 1;
		}
}

void Mesh3DStorage::append(const Mesh3DStorage &other)
{
	if (other.type != type)
		type = MeshType::HYB;

	const int n_v = n_vertices();
	const int n_f = n_faces();

	assert(points.rows() == other.points.rows());
	points.conservativeResize(points.rows(), n_v + other.points.cols());
	points.rightCols(other.points.cols()) = other.points;

	face_vs.append(other.face_vs, n_v);
	face_boundary.insert(face_boundary.end(), other.face_boundary.begin(), other.face_boundary.end());

	cell_vs.append(other.cell_vs, n_v);
	cell_input_vs.append(other.cell_input_vs, n_v);
	cell_fs.append(other.cell_fs, n_f);
	cell_fs_flag.append(other.cell_fs_flag, 0);
	cell_hex.insert(cell_hex.end(), other.cell_hex.begin(), other.cell_hex.end());
	cell_kernel.insert(cell_kernel.end(), other.cell_kernel.begin(), other.cell_kernel.end());
}

size_t Mesh3DStorage::memory() const
{
	return points.size() * sizeof(double)
		   + vertex_vs.memory() + vertex_es.memory() + vertex_fs.memory() + vertex_hs.memory()
		   + vertex_boundary.capacity()
		   + edge_vs.capacity() * sizeof(std::array<uint32_t, 2>) + edge_fs.memory() + edge_hs.memory() + edge_boundary.capacity()
		   + face_vs.memory() + face_es.memory() + face_hs.memory() + face_hs_slot.memory() + face_boundary.capacity()
		   + cell_vs.memory() + cell_input_vs.memory() + cell_es.memory() + cell_fs.memory() + cell_fs_flag.memory()
		   + cell_hex.capacity() + cell_kernel.capacity() * sizeof(std::array<double, 3>);
}

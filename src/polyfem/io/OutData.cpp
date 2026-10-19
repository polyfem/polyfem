#include "OutData.hpp"

#include <array>
#include <map>

#include "Evaluator.hpp"
#include "MatrixIO.hpp"

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/AssemblyValues.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/assembler/Mass.hpp>

#include <polyfem/basis/ElementBases.hpp>

#include <polyfem/mesh/Obstacle.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/utils/getRSS.h>
#include <polyfem/utils/EdgeSampler.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/par_for.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>
#include <polyfem/autogen/prism_bases.hpp>
#include <polyfem/autogen/auto_pyramid_bases.hpp>

#include <paraviewo/VTMWriter.hpp>
#include <paraviewo/PVDWriter.hpp>

#include <ipc/potentials/normal_adhesion_potential.hpp>
#include <ipc/potentials/tangential_adhesion_potential.hpp>

#include <SimpleBVH/BVH.hpp>

#include <igl/write_triangle_mesh.h>
#include <igl/edges.h>
#include <igl/facet_adjacency_matrix.h>
#include <igl/connected_components.h>

#include <ipc/ipc.hpp>

#include <algorithm>
#include <cmath>

#include <filesystem>

namespace polyfem::io
{
	bool OutputFieldOptions::export_field(const std::string &field) const
	{
		return fields.empty() || std::find(fields.begin(), fields.end(), field) != fields.end();
	}

	using CellType = paraviewo::CellType;
	using CellElement = paraviewo::CellElement;

	namespace
	{
		void add_output_fields(
			paraviewo::ParaviewWriter &writer,
			const OutputSample &sample,
			const OutputFieldFunction &output_fields)
		{
			if (!output_fields)
				return;

			for (const OutputField &field : output_fields(sample))
			{
				if (field.values.rows() <= 0)
					continue;

				const int expected_rows =
					field.association == OutputField::Association::Cell
						? sample.cell_count
						: sample.points.rows();
				if (field.values.rows() != expected_rows)
				{
					logger().warn(
						"Skipping output field '{}' with {} rows; expected {} {} rows",
						field.name, field.values.rows(), expected_rows,
						field.association == OutputField::Association::Cell ? "cell" : "point");
					continue;
				}

				if (field.association == OutputField::Association::Cell)
					writer.add_cell_field(field.name, field.values);
				else
					writer.add_field(field.name, field.values);
			}
		}

		void avoid_pyramid_apex(Eigen::MatrixXd &points)
		{
			assert(points.cols() == 3);
			constexpr double eps = 1e-8;
			for (int i = 0; i < points.rows(); ++i)
			{
				if (std::abs(points(i, 2) - 1.0) < eps)
					points(i, 2) = 1.0 - eps;
			}
		}

		void pyramid_nodes_for_output(const int order, Eigen::MatrixXd &points)
		{
			autogen::pyramid_nodes_3d(order, points);
			avoid_pyramid_apex(points);
		}

		// ------------------------------------------------------------------
		// helpers for the hybrid (prism/pyramid) collision-proxy paths
		// ------------------------------------------------------------------

		// face/edge parameters are fractions k/n with n <= 8, so coordinates
		// in units of 1/PROXY_PARAM_SCALE are exact integers (lcm of 1..8)
		constexpr long PROXY_PARAM_SCALE = 840;

		int prism_q_order(const basis::ElementBases &b)
		{
			const int p = b.bases.empty() ? 1 : b.bases.front().order();
			const int n_tri = (p + 1) * (p + 2) / 2;
			const int q = (n_tri > 0 && b.bases.size() % n_tri == 0) ? int(b.bases.size()) / n_tri - 1 : p;
			return std::max(1, q);
		}

		// reference-node layout of an element (vertices first); false if unsupported
		bool element_ref_nodes(const mesh::Mesh &mesh, const int el, const basis::ElementBases &b, Eigen::MatrixXd &ref_nodes)
		{
			const int p = b.bases.empty() ? 1 : b.bases.front().order();
			if (mesh.is_simplex(el))
				autogen::p_nodes_3d(p, ref_nodes);
			else if (mesh.is_cube(el))
				autogen::q_nodes_3d(p, ref_nodes);
			else if (mesh.is_prism(el))
				autogen::prism_nodes_3d(p, prism_q_order(b), ref_nodes);
			else if (mesh.is_pyramid(el))
				autogen::pyramid_nodes_3d(p, ref_nodes);
			else
				return false;
			return ref_nodes.rows() == long(b.bases.size());
		}

		int element_n_vertices(const mesh::Mesh &mesh, const int el)
		{
			if (mesh.is_simplex(el))
				return 4;
			if (mesh.is_pyramid(el))
				return 5;
			if (mesh.is_prism(el))
				return 6;
			assert(mesh.is_cube(el));
			return 8;
		}

		// local vertex pairs forming the element's edges, derived from the
		// reference vertex coordinates (no vertex-ordering assumptions)
		std::vector<std::pair<int, int>> element_ref_edges(const mesh::Mesh &mesh, const int el, const Eigen::MatrixXd &ref_nodes)
		{
			const int nv = element_n_vertices(mesh, el);
			const auto n_coord_diffs = [&](int a, int b) {
				int d = 0;
				for (int c = 0; c < 3; ++c)
					if (std::abs(ref_nodes(a, c) - ref_nodes(b, c)) > 1e-12)
						++d;
				return d;
			};
			int apex = -1;
			if (mesh.is_pyramid(el))
				for (int a = 0; a < nv; ++a)
					if (std::abs(ref_nodes(a, 2) - 1.0) < 1e-12)
						apex = a;

			std::vector<std::pair<int, int>> edges;
			for (int a = 0; a < nv; ++a)
			{
				for (int b = a + 1; b < nv; ++b)
				{
					bool is_edge;
					if (mesh.is_simplex(el)) // every vertex pair
						is_edge = true;
					else if (mesh.is_prism(el)) // triangle x [0,1]
						is_edge = std::abs(ref_nodes(a, 2) - ref_nodes(b, 2)) < 1e-12
								  || (std::abs(ref_nodes(a, 0) - ref_nodes(b, 0)) < 1e-12
									  && std::abs(ref_nodes(a, 1) - ref_nodes(b, 1)) < 1e-12);
					else if (mesh.is_pyramid(el)) // base square + base-to-apex
						is_edge = (a == apex || b == apex) || n_coord_diffs(a, b) == 1;
					else // hex
						is_edge = n_coord_diffs(a, b) == 1;
					if (is_edge)
						edges.emplace_back(a, b);
				}
			}
			return edges;
		}

		// For every mesh edge, keyed by its endpoint DOFs (smaller first): the
		// owned edge-interior DOFs located on it as (parameter from the
		// smaller-DOF endpoint in units of 1/PROXY_PARAM_SCALE, dof, node
		// position), sorted by parameter.  Built from ALL elements: an edge on
		// the domain boundary can get its DOFs from an interior element (e.g. a
		// prism wedged between two promoted tets).  Stitched nodes
		// (glob.size() != 1, no DOF of their own) are not registered.
		using EdgeDofs = std::vector<std::tuple<long, int, Eigen::Vector3d>>;
		std::map<std::pair<int, int>, EdgeDofs> build_edge_dofs(const mesh::Mesh &mesh, const std::vector<basis::ElementBases> &bases)
		{
			std::map<std::pair<int, int>, EdgeDofs> edge_dofs;
			Eigen::MatrixXd ref_nodes;
			for (int el = 0; el < int(bases.size()); ++el)
			{
				const basis::ElementBases &b = bases[el];
				if (b.bases.empty() || !element_ref_nodes(mesh, el, b, ref_nodes))
					continue;

				const int nv = element_n_vertices(mesh, el);
				std::vector<int> vd(nv, -1);
				bool ok = true;
				for (int i = 0; i < nv; ++i)
				{
					const auto &glob = b.bases[i].global();
					assert(glob.size() == 1); // vertices always own their DOF
					if (glob.size() != 1)
					{
						ok = false;
						break;
					}
					vd[i] = glob.front().index;
				}
				if (!ok)
					continue;

				const auto edges = element_ref_edges(mesh, el, ref_nodes);
				for (int j = nv; j < int(b.bases.size()); ++j)
				{
					const auto &glob = b.bases[j].global();
					if (glob.size() != 1)
						continue;
					const Eigen::RowVector3d r = ref_nodes.row(j);
					for (const auto &e : edges)
					{
						const Eigen::RowVector3d pa = ref_nodes.row(e.first);
						const Eigen::RowVector3d d = ref_nodes.row(e.second) - pa;
						const double t = (r - pa).dot(d) / d.squaredNorm();
						if (t < 1e-9 || t > 1 - 1e-9 || ((r - pa) - t * d).norm() > 1e-9)
							continue; // not on this edge
						long tl = std::lround(t * PROXY_PARAM_SCALE);
						assert(std::abs(t * PROXY_PARAM_SCALE - tl) < 1e-6);
						int va = vd[e.first], vb = vd[e.second];
						if (va > vb)
						{
							std::swap(va, vb);
							tl = PROXY_PARAM_SCALE - tl;
						}
						EdgeDofs &ed = edge_dofs[{va, vb}];
						const int dof = glob.front().index;
						bool present = false;
						for (const auto &existing : ed)
							if (std::get<1>(existing) == dof)
							{
								assert(std::get<0>(existing) == tl);
								present = true;
								break;
							}
						if (!present)
							ed.emplace_back(tl, dof, glob.front().node.transpose());
						break;
					}
				}
			}
			for (auto &kv : edge_dofs)
				std::sort(kv.second.begin(), kv.second.end(),
						  [](const EdgeDofs::value_type &x, const EdgeDofs::value_type &y) { return std::get<0>(x) < std::get<0>(y); });
			return edge_dofs;
		}

		// Structured triangulation for faces whose points form a complete
		// regular lattice (all faces away from mixed-order interfaces).  The
		// row-wise patterns keep vertex valence balanced -- the lexicographic
		// sweep below concentrates fan triangles on lex-extreme corners, which
		// at order 3+ can push one-ring sizes past the smooth-contact
		// N_VERT_NEIGHBORS_3D cap.  Returns false if the points are not a
		// complete lattice (caller falls back to the sweep).
		bool triangulate_lattice(const std::vector<std::array<long, 2>> &pts, const int nfv, std::vector<std::array<int, 3>> &tris)
		{
			constexpr long S = PROXY_PARAM_SCALE;
			const int n = int(pts.size());
			std::map<std::array<long, 2>, int> id;
			for (int i = 0; i < n; ++i)
				if (!id.emplace(pts[i], i).second)
					return false;
			tris.clear();

			if (nfv == 3)
			{
				// n == (k+1)(k+2)/2 for a complete triangular lattice of order k
				const int k = int(std::lround((std::sqrt(8.0 * n + 1.0) - 3.0) / 2.0));
				if ((k + 1) * (k + 2) / 2 != n || k < 1 || S % k != 0)
					return false;
				const long h = S / k;
				const auto at = [&](int i, int j) -> int {
					const auto it = id.find({{i * h, j * h}});
					return it == id.end() ? -1 : it->second;
				};
				for (int j = 0; j <= k; ++j)
					for (int i = 0; i <= k - j; ++i)
						if (at(i, j) < 0)
							return false;
				for (int j = 0; j < k; ++j)
				{
					for (int i = 0; i < k - j; ++i)
					{
						tris.push_back({{at(i, j), at(i + 1, j), at(i, j + 1)}});
						if (i + j < k - 1)
							tris.push_back({{at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)}});
					}
				}
				return true;
			}

			// quad: complete (k1+1) x (k2+1) tensor grid, possibly anisotropic
			std::set<long> us, vs;
			for (const auto &p : pts)
			{
				us.insert(p[0]);
				vs.insert(p[1]);
			}
			const int k1 = int(us.size()) - 1, k2 = int(vs.size()) - 1;
			if (k1 < 1 || k2 < 1 || (k1 + 1) * (k2 + 1) != n || S % k1 != 0 || S % k2 != 0)
				return false;
			const long h1 = S / k1, h2 = S / k2;
			const auto at = [&](int i, int j) -> int {
				const auto it = id.find({{i * h1, j * h2}});
				return it == id.end() ? -1 : it->second;
			};
			for (int j = 0; j <= k2; ++j)
				for (int i = 0; i <= k1; ++i)
					if (at(i, j) < 0)
						return false;
			for (int j = 0; j < k2; ++j)
			{
				for (int i = 0; i < k1; ++i)
				{
					tris.push_back({{at(i, j), at(i + 1, j), at(i, j + 1)}});
					tris.push_back({{at(i + 1, j + 1), at(i, j + 1), at(i + 1, j)}});
				}
			}
			return true;
		}

		// Triangulate a 2D point set in convex position (boundary points lie on
		// the convex hull, possibly collinear; interior points allowed) into
		// strictly positive-area CCW triangles via an incremental lexicographic
		// sweep.  Collinear boundary chains are preserved -- no triangle edge
		// ever spans a boundary point -- which is what keeps neighboring faces
		// of the proxy conforming.  Coordinates must be exact integers.
		void triangulate_convex_pointset(const std::vector<std::array<long, 2>> &pts, std::vector<std::array<int, 3>> &tris)
		{
			const int n = int(pts.size());
			tris.clear();
			std::vector<int> order(n);
			for (int i = 0; i < n; ++i)
				order[i] = i;
			std::sort(order.begin(), order.end(), [&](int a, int b) { return pts[a] < pts[b]; });
			const auto orient = [&](int a, int b, int c) -> long long {
				return (long long)(pts[b][0] - pts[a][0]) * (pts[c][1] - pts[a][1])
					   - (long long)(pts[b][1] - pts[a][1]) * (pts[c][0] - pts[a][0]);
			};

			// initial (possibly collinear) chain
			std::vector<int> hull;
			int k = 0;
			for (; k < n; ++k)
			{
				if (hull.size() >= 2 && orient(hull[0], hull[1], order[k]) != 0)
					break;
				hull.push_back(order[k]);
			}
			if (k == n)
				return; // all points collinear: nothing to emit

			// attach the first off-line point: fan over the chain, make hull CCW
			{
				const int p = order[k];
				const bool left = orient(hull[0], hull[1], p) > 0;
				for (int i = 0; i + 1 < int(hull.size()); ++i)
				{
					if (left)
						tris.push_back({{hull[i], hull[i + 1], p}});
					else
						tris.push_back({{hull[i + 1], hull[i], p}});
				}
				if (!left)
					std::reverse(hull.begin(), hull.end());
				hull.push_back(p);
				++k;
			}

			for (; k < n; ++k)
			{
				const int p = order[k];
				const int m = int(hull.size());
				// visible hull edges: p strictly right of a->b on the CCW cycle
				std::vector<bool> vis(m);
				for (int i = 0; i < m; ++i)
					vis[i] = orient(hull[i], hull[(i + 1) % m], p) < 0;
				int s = 0;
				while (s < m && !(vis[s] && !vis[(s + m - 1) % m]))
					++s;
				assert(s < m); // p is outside the hull (lexicographic order)
				int e = s;
				while (vis[e % m])
				{
					// orient(a, b, p) < 0  =>  (a, p, b) is CCW
					tris.push_back({{hull[e % m], p, hull[(e + 1) % m]}});
					++e;
				}
				// replace the strictly-visible part of the hull with p
				std::vector<int> new_hull;
				new_hull.push_back(p);
				for (int i = e % m; i != s; i = (i + 1) % m)
					new_hull.push_back(hull[i]);
				new_hull.push_back(hull[s]);
				hull.swap(new_hull);
			}
		}

	} // namespace

	void OutGeometryData::extract_boundary_mesh_sampled(
		const mesh::Mesh &mesh,
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		Eigen::MatrixXd &node_positions,
		Eigen::MatrixXi &boundary_edges,
		Eigen::MatrixXi &boundary_triangles,
		std::vector<Eigen::Triplet<double>> &displacement_map_entries,
		const int sampling_order)
	{
		using namespace polyfem::mesh;

		// Sample every boundary face on a uniform lattice of the globally
		// maximal order M through the element bases: the proxy is conforming
		// and watertight regardless of order mismatches across interfaces
		// (every shared edge is split into M chords on both sides), stitched
		// interface nodes are handled intrinsically by the basis evaluation,
		// and the displacement map carries each proxy vertex's (possibly
		// weighted) dependence on the real DOFs.
		if (!mesh.is_volume() || mesh.has_poly()
			|| !dynamic_cast<const Mesh3D &>(mesh).is_conforming())
		{
			logger().warn("max_order collision-proxy sampling requires a conforming volume mesh without polytopes; falling back to the standard boundary extraction");
			extract_boundary_mesh(mesh, n_bases, bases, total_local_boundary,
								  node_positions, boundary_edges, boundary_triangles, displacement_map_entries);
			return;
		}

		displacement_map_entries.clear();
		const Mesh3D &mesh3d = dynamic_cast<const Mesh3D &>(mesh);

		int M = 1;
		if (sampling_order > 0)
			M = sampling_order; // explicit lattice order (finer OR coarser than the elements)
		else
			for (const LocalBoundary &lb : total_local_boundary)
			{
				const basis::ElementBases &b = bases[lb.element_id()];
				if (b.bases.empty())
					continue;
				int o = b.bases.front().order();
				if (mesh.is_prism(lb.element_id()))
					o = std::max(o, prism_q_order(b));
				M = std::max(M, o);
			}

		// samples shared between neighboring faces merge combinatorially, by
		// the mesh primitive they lie on -- {0, vertex_id} for face corners,
		// {1, vmin, vmax, step-from-vmin} for edge samples (every face uses
		// the same global lattice order M, so steps coincide exactly), and
		// {2, face_id, i, r} for face-interior samples (a boundary face
		// belongs to exactly one element) -- no floating-point tolerance
		std::map<std::array<int, 4>, int> vertex_id;
		std::vector<Eigen::Vector3d> vertices;
		std::vector<std::tuple<int, int, int>> proxy_tris;

		for (const LocalBoundary &lb : total_local_boundary)
		{
			const int el = lb.element_id();
			const basis::ElementBases &b = bases[el];
			Eigen::MatrixXd ref_nodes;
			if (b.bases.empty() || !element_ref_nodes(mesh, el, b, ref_nodes))
				continue;

			// the rational pyramid bases are 0/0 at the apex (z=1), which is a
			// corner of every triangular pyramid face; apex samples are instead
			// evaluated below via the nodal limit (apex basis 1, all others 0)
			int apex_node = -1;
			if (mesh.is_pyramid(el))
				for (int i = 0; i < ref_nodes.rows(); ++i)
					if (std::abs(ref_nodes(i, 2) - 1.0) < 1e-12)
					{
						apex_node = i;
						break;
					}

			for (int j = 0; j < lb.size(); ++j)
			{
				const int eid = lb.global_primitive_id(j);
				const int nfv = mesh3d.n_face_vertices(eid);
				assert(nfv == 3 || nfv == 4);

				Navigation3D::Index nav;
				for (int lf = 0; lf < mesh3d.n_cell_faces(el); ++lf)
				{
					nav = mesh3d.get_index_from_element(el, lf, 0);
					if (nav.face == eid)
						break;
				}
				assert(nav.face == eid);
				std::vector<int> gv(nfv);
				{
					Navigation3D::Index cur = nav;
					for (int k = 0; k < nfv; ++k)
					{
						gv[k] = cur.vertex;
						cur = mesh3d.next_around_face(cur);
					}
				}

				// Map the global face vertices to canonical element-reference
				// vertices. This cannot use the first entries of `nodes`: nodal
				// Lagrange faces store corners first, but spline faces use tensor-grid
				// ordering (where the first four entries are generally not corners).
				Eigen::MatrixXd ref_vertices;
				std::vector<int> local_to_global;
				if (mesh.is_simplex(el))
				{
					autogen::p_nodes_3d(1, ref_vertices);
					const auto vertices = mesh3d.get_ordered_vertices_from_tet(el);
					local_to_global.assign(vertices.begin(), vertices.end());
				}
				else if (mesh.is_cube(el))
				{
					autogen::q_nodes_3d(1, ref_vertices);
					const auto vertices = mesh3d.get_ordered_vertices_from_hex(el);
					local_to_global.assign(vertices.begin(), vertices.end());
				}
				else if (mesh.is_prism(el))
				{
					autogen::prism_nodes_3d(1, 1, ref_vertices);
					const auto vertices = mesh3d.get_ordered_vertices_from_prism(el);
					local_to_global.assign(vertices.begin(), vertices.end());
				}
				else
				{
					assert(mesh.is_pyramid(el));
					autogen::pyramid_nodes_3d(1, ref_vertices);
					const auto vertices = mesh3d.get_ordered_vertices_from_pyramid(el);
					local_to_global.assign(vertices.begin(), vertices.end());
				}

				std::vector<Eigen::RowVector3d> c(nfv);
				for (int k = 0; k < nfv; ++k)
				{
					const auto it = std::find(local_to_global.begin(), local_to_global.end(), gv[k]);
					assert(it != local_to_global.end());
					c[k] = ref_vertices.row(std::distance(local_to_global.begin(), it));
				}

				Eigen::MatrixXd pts;
				std::vector<std::array<int, 2>> coords;
				std::vector<std::tuple<int, int, int>> local_tris;
				if (nfv == 3)
				{
					pts.resize((M + 1) * (M + 2) / 2, 3);
					coords.resize(pts.rows());
					std::vector<int> off(M + 2, 0);
					for (int r = 0; r <= M; ++r)
						off[r + 1] = off[r] + (M + 1 - r);
					for (int r = 0; r <= M; ++r)
						for (int i = 0; i <= M - r; ++i)
						{
							pts.row(off[r] + i) = c[0] + (double(i) / M) * (c[1] - c[0]) + (double(r) / M) * (c[2] - c[0]);
							coords[off[r] + i] = {{i, r}};
						}
					for (int r = 0; r < M; ++r)
					{
						for (int i = 0; i < M - r; ++i)
						{
							local_tris.emplace_back(off[r] + i, off[r] + i + 1, off[r + 1] + i);
							if (i + r < M - 1)
								local_tris.emplace_back(off[r] + i + 1, off[r + 1] + i + 1, off[r + 1] + i);
						}
					}
				}
				else
				{
					pts.resize((M + 1) * (M + 1), 3);
					coords.resize(pts.rows());
					const auto gid = [M](const int i, const int r) { return r * (M + 1) + i; };
					for (int r = 0; r <= M; ++r)
					{
						for (int i = 0; i <= M; ++i)
						{
							const double u = double(i) / M, v = double(r) / M;
							pts.row(gid(i, r)) = (1 - u) * (1 - v) * c[0] + u * (1 - v) * c[1] + u * v * c[2] + (1 - u) * v * c[3];
							coords[gid(i, r)] = {{i, r}};
						}
					}
					for (int r = 0; r < M; ++r)
					{
						for (int i = 0; i < M; ++i)
						{
							local_tris.emplace_back(gid(i, r), gid(i + 1, r), gid(i, r + 1));
							local_tris.emplace_back(gid(i + 1, r + 1), gid(i, r + 1), gid(i + 1, r));
						}
					}
				}

				std::vector<polyfem::assembler::AssemblyValues> vals;
				b.evaluate_bases(pts, vals);

				const auto edge_key = [M](const int va, const int vb, const int step) {
					return va < vb ? std::array<int, 4>{{1, va, vb, step}}
								   : std::array<int, 4>{{1, vb, va, M - step}};
				};

				std::vector<int> ids(pts.rows());
				for (int s = 0; s < pts.rows(); ++s)
				{
					const int i = coords[s][0], r = coords[s][1];
					std::array<int, 4> key;
					if (nfv == 3)
					{
						if (r == 0 && i == 0)
							key = {{0, gv[0], 0, 0}};
						else if (r == 0 && i == M)
							key = {{0, gv[1], 0, 0}};
						else if (r == M)
							key = {{0, gv[2], 0, 0}};
						else if (r == 0)
							key = edge_key(gv[0], gv[1], i);
						else if (i == 0)
							key = edge_key(gv[0], gv[2], r);
						else if (i + r == M)
							key = edge_key(gv[1], gv[2], r);
						else
							key = {{2, eid, i, r}};
					}
					else
					{
						if (i == 0 && r == 0)
							key = {{0, gv[0], 0, 0}};
						else if (i == M && r == 0)
							key = {{0, gv[1], 0, 0}};
						else if (i == M && r == M)
							key = {{0, gv[2], 0, 0}};
						else if (i == 0 && r == M)
							key = {{0, gv[3], 0, 0}};
						else if (r == 0)
							key = edge_key(gv[0], gv[1], i);
						else if (i == M)
							key = edge_key(gv[1], gv[2], r);
						else if (r == M)
							key = edge_key(gv[3], gv[2], i);
						else if (i == 0)
							key = edge_key(gv[0], gv[3], r);
						else
							key = {{2, eid, i, r}};
					}

					const auto it = vertex_id.find(key);
					if (it != vertex_id.end())
					{
						ids[s] = it->second;
						continue;
					}

					Eigen::Vector3d pos = Eigen::Vector3d::Zero();
					std::map<int, double> weights;
					if (apex_node >= 0 && std::abs(pts(s, 2) - 1.0) < 1e-12)
					{
						for (const auto &g : b.bases[apex_node].global())
						{
							pos += g.val * g.node.transpose();
							weights[g.index] += g.val;
						}
					}
					else
						for (size_t i2 = 0; i2 < vals.size(); ++i2)
						{
							const double Ni = vals[i2].val(s);
							if (std::abs(Ni) < 1e-12)
								continue;
							for (const auto &g : b.bases[i2].global())
							{
								pos += Ni * g.val * g.node.transpose();
								weights[g.index] += Ni * g.val;
							}
						}
					assert(pos.allFinite());

					const int vid = int(vertices.size());
					vertex_id[key] = vid;
					vertices.push_back(pos);
					for (const auto &kv : weights)
						if (std::abs(kv.second) > 1e-10)
							displacement_map_entries.emplace_back(vid, kv.first, kv.second);
					ids[s] = vid;
				}

				for (const auto &t : local_tris)
					proxy_tris.emplace_back(ids[std::get<0>(t)], ids[std::get<1>(t)], ids[std::get<2>(t)]);
			}
		}

		node_positions.resize(vertices.size(), 3);
		for (int i = 0; i < int(vertices.size()); ++i)
			node_positions.row(i) = vertices[i];

		boundary_triangles.resize(proxy_tris.size(), 3);
		for (int i = 0; i < int(proxy_tris.size()); ++i)
			boundary_triangles.row(i) << std::get<0>(proxy_tris[i]), std::get<2>(proxy_tris[i]), std::get<1>(proxy_tris[i]);

		if (boundary_triangles.rows() > 0)
			igl::edges(boundary_triangles, boundary_edges);

		if (const char *dump = getenv("POLYFEM_DUMP_COLLISION_PROXY"))
			igl::write_triangle_mesh(dump, node_positions, boundary_triangles);
	}

	void OutGeometryData::extract_boundary_mesh(
		const mesh::Mesh &mesh,
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		Eigen::MatrixXd &node_positions,
		Eigen::MatrixXi &boundary_edges,
		Eigen::MatrixXi &boundary_triangles,
		std::vector<Eigen::Triplet<double>> &displacement_map_entries)
	{
		using namespace polyfem::mesh;

		displacement_map_entries.clear();

		if (mesh.is_volume())
		{
			if (mesh.has_poly())
			{
				logger().warn("Skipping as the mesh has polygons");
				return;
			}

			const bool is_simplicial = mesh.is_simplicial();

			std::vector<Eigen::Vector3d> node_positions_vec;
			node_positions_vec.reserve(n_bases + (is_simplicial ? 0 : mesh.n_faces()));

			// node_positions.resize(n_bases + (is_simplicial ? 0 : mesh.n_faces()), 3);
			// node_positions.setZero();
			const Mesh3D &mesh3d = dynamic_cast<const Mesh3D &>(mesh);

			std::vector<std::tuple<int, int, int>> tris;

			std::vector<bool> visited_node(n_bases, false);

			std::stringstream print_warning;

			// Hybrid (prism/pyramid) meshes: element orders can differ across an
			// interface (anisotropic prisms promote their tet/pyramid neighbors),
			// so tessellating each boundary face at its own order produces
			// T-junctions along shared edges, and stitched interface nodes have
			// no DOF of their own.  Build the proxy from the DOFs on the
			// boundary instead: every proxy vertex IS a global DOF (weight-1
			// displacement map), each face edge is subdivided by the DOFs
			// located on it -- a property of the edge, not of the element
			// looking at it, so neighboring faces conform by construction --
			// and face interiors by the element's own owned face nodes.
			// Stitched nodes are skipped; the DOFs they depend on subdivide the
			// edge instead.
			bool has_prism_or_pyramid = false;
			for (const LocalBoundary &lb : total_local_boundary)
			{
				if (mesh.is_prism(lb.element_id()) || mesh.is_pyramid(lb.element_id()))
				{
					has_prism_or_pyramid = true;
					break;
				}
			}

			if (has_prism_or_pyramid && mesh3d.is_conforming())
			{
				const auto edge_dofs = build_edge_dofs(mesh, bases);
				constexpr long S = PROXY_PARAM_SCALE;

				const auto emit_dof = [&](const int gindex, const Eigen::Vector3d &pos) {
					if (gindex >= int(node_positions_vec.size()))
						node_positions_vec.resize(gindex + 1, Eigen::Vector3d::Zero());
					node_positions_vec[gindex] = pos;
					if (!visited_node[gindex])
						displacement_map_entries.emplace_back(gindex, gindex, 1);
					visited_node[gindex] = true;
				};

				Eigen::MatrixXd ref_nodes;
				for (const LocalBoundary &lb : total_local_boundary)
				{
					const int el = lb.element_id();
					const basis::ElementBases &b = bases[el];
					if (b.bases.empty() || !element_ref_nodes(mesh, el, b, ref_nodes))
						continue;

					for (int j = 0; j < lb.size(); ++j)
					{
						const int eid = lb.global_primitive_id(j);
						const Eigen::VectorXi nodes = b.local_nodes_for_primitive(eid, mesh3d);
						const int nfv = mesh3d.n_face_vertices(eid);
						assert(nfv == 3 || nfv == 4);
						assert(nodes.size() >= nfv);

						// exact face parameters (units of 1/S) and the DOF of each point
						std::vector<std::array<long, 2>> pts;
						std::vector<int> dof;

						// face corners come first in the local ordering (cyclic for quads)
						std::array<std::array<long, 2>, 4> cp;
						if (nfv == 3)
							cp = {{{{0, 0}}, {{S, 0}}, {{0, S}}, {{0, 0}}}};
						else
							cp = {{{{0, 0}}, {{S, 0}}, {{S, S}}, {{0, S}}}};
						std::array<int, 4> cd{{-1, -1, -1, -1}};
						bool ok = true;
						for (int k = 0; k < nfv; ++k)
						{
							const auto &glob = b.bases[nodes(k)].global();
							assert(glob.size() == 1); // face corners always own their DOF
							if (glob.size() != 1)
							{
								ok = false;
								break;
							}
							cd[k] = glob.front().index;
							pts.push_back(cp[k]);
							dof.push_back(cd[k]);
							emit_dof(cd[k], glob.front().node.transpose());
						}
						if (!ok)
							continue;

						// edge subdivision: the DOFs located on each face edge
						for (int k = 0; k < nfv; ++k)
						{
							const int va = cd[k], vb = cd[(k + 1) % nfv];
							const auto it = edge_dofs.find({std::min(va, vb), std::max(va, vb)});
							if (it == edge_dofs.end())
								continue;
							for (const auto &ed : it->second)
							{
								const long s = va < vb ? std::get<0>(ed) : S - std::get<0>(ed);
								const auto &A = cp[k];
								const auto &B = cp[(k + 1) % nfv];
								pts.push_back({{(A[0] * (S - s) + B[0] * s) / S, (A[1] * (S - s) + B[1] * s) / S}});
								dof.push_back(std::get<1>(ed));
								emit_dof(std::get<1>(ed), std::get<2>(ed));
							}
						}

						// owned face-interior nodes at the element's own resolution
						// (reference faces are planar parallelograms -> affine map)
						const Eigen::RowVector3d c0 = ref_nodes.row(nodes(0));
						const Eigen::RowVector3d A3 = ref_nodes.row(nodes(1)) - c0;
						const Eigen::RowVector3d B3 = ref_nodes.row(nodes(nfv - 1)) - c0;
						const double aa = A3.squaredNorm(), bb = B3.squaredNorm(), ab = A3.dot(B3);
						const double det = aa * bb - ab * ab;
						for (long n = nfv; n < nodes.size(); ++n)
						{
							const auto &glob = b.bases[nodes(n)].global();
							if (glob.size() != 1)
								continue; // stitched interface node
							const Eigen::RowVector3d d3 = ref_nodes.row(nodes(n)) - c0;
							const double du = d3.dot(A3), dv = d3.dot(B3);
							const double u = (du * bb - dv * ab) / det;
							const double v = (dv * aa - du * ab) / det;
							const long lu = std::lround(u * S), lv = std::lround(v * S);
							assert(std::abs(u * S - lu) < 1e-6 && std::abs(v * S - lv) < 1e-6);
							// nodes on a face edge already subdivide the edge above
							const bool on_edge = nfv == 3
													 ? (lu == 0 || lv == 0 || lu + lv == S)
													 : (lu == 0 || lu == S || lv == 0 || lv == S);
							if (on_edge)
								continue;
							pts.push_back({{lu, lv}});
							dof.push_back(glob.front().index);
							emit_dof(glob.front().index, glob.front().node.transpose());
						}

						std::vector<std::array<int, 3>> local_tris;
						if (!triangulate_lattice(pts, nfv, local_tris))
							triangulate_convex_pointset(pts, local_tris);
						for (const auto &t : local_tris)
							tris.emplace_back(dof[t[0]], dof[t[1]], dof[t[2]]);
					}
				}

				// downstream consumers (e.g. the shape-derivative code) expect every
				// FE node to have a row, as the pre-split extraction guaranteed
				node_positions_vec.resize(
					std::max(node_positions_vec.size(), size_t(n_bases)), Eigen::Vector3d::Zero());

				node_positions.resize(node_positions_vec.size(), 3);
				for (int i = 0; i < int(node_positions_vec.size()); ++i)
					node_positions.row(i) = node_positions_vec[i];

				boundary_triangles.resize(tris.size(), 3);
				for (int i = 0; i < int(tris.size()); ++i)
					boundary_triangles.row(i) << std::get<0>(tris[i]), std::get<2>(tris[i]), std::get<1>(tris[i]);

				if (boundary_triangles.rows() > 0)
					igl::edges(boundary_triangles, boundary_edges);

				if (const char *dump = getenv("POLYFEM_DUMP_COLLISION_PROXY"))
					igl::write_triangle_mesh(dump, node_positions, boundary_triangles);

				return;
			}

			for (const LocalBoundary &lb : total_local_boundary)
			{
				const basis::ElementBases &b = bases[lb.element_id()];

				for (int j = 0; j < lb.size(); ++j)
				{
					const int eid = lb.global_primitive_id(j);
					const int lid = lb[j];
					const Eigen::VectorXi nodes = b.local_nodes_for_primitive(eid, mesh3d);

					if (mesh.is_cube(lb.element_id()))
					{
						assert(!is_simplicial);
						assert(!mesh.has_poly());
						std::vector<int> loc_nodes;
						RowVectorNd bary = RowVectorNd::Zero(3);

						for (long n = 0; n < nodes.size(); ++n)
						{
							auto &bs = b.bases[nodes(n)];
							const auto &glob = bs.global();
							if (glob.size() != 1)
								continue;

							int gindex = glob.front().index;
							node_positions_vec.resize(std::max(int(node_positions_vec.size()), gindex + 1));
							node_positions_vec[gindex] = glob.front().node;
							bary += glob.front().node;
							loc_nodes.push_back(gindex);
						}

						if (loc_nodes.size() != 4)
						{
							logger().trace("skipping element {} since it is not Q1", eid);
							continue;
						}

						bary /= 4;

						const int new_node = n_bases + eid;
						node_positions_vec.resize(std::max(int(node_positions_vec.size()), new_node + 1));
						node_positions_vec[new_node] = bary;
						tris.emplace_back(loc_nodes[1], loc_nodes[0], new_node);
						tris.emplace_back(loc_nodes[2], loc_nodes[1], new_node);
						tris.emplace_back(loc_nodes[3], loc_nodes[2], new_node);
						tris.emplace_back(loc_nodes[0], loc_nodes[3], new_node);

						for (int q = 0; q < 4; ++q)
						{
							if (!visited_node[loc_nodes[q]])
								displacement_map_entries.emplace_back(loc_nodes[q], loc_nodes[q], 1);

							visited_node[loc_nodes[q]] = true;
							displacement_map_entries.emplace_back(new_node, loc_nodes[q], 0.25);
						}

						continue;
					}
					else if (mesh.is_prism(lb.element_id()))
					{
						assert(!is_simplicial);
						assert(!mesh.has_poly());
						std::vector<int> loc_nodes;
						std::vector<int> loc_local_nodes;

						for (long n = 0; n < nodes.size(); ++n)
						{
							auto &bs = b.bases[nodes(n)];
							const auto &glob = bs.global();
							if (glob.size() != 1)
								continue;

							int gindex = glob.front().index;
							node_positions_vec.resize(std::max(int(node_positions_vec.size()), gindex + 1));
							node_positions_vec[gindex] = glob.front().node;
							loc_nodes.push_back(gindex);
							loc_local_nodes.push_back(nodes(n));
						}

						auto update_mapping = [&displacement_map_entries, &visited_node](const std::vector<int> &loc_nodes) {
							for (int k = 0; k < loc_nodes.size(); ++k)
							{
								if (!visited_node[loc_nodes[k]])
									displacement_map_entries.emplace_back(loc_nodes[k], loc_nodes[k], 1);

								visited_node[loc_nodes[k]] = true;
							}
						};

						// tri face
						if (lid < 2)
						{
							if (loc_nodes.size() == 3)
							{
								tris.emplace_back(loc_nodes[0], loc_nodes[1], loc_nodes[2]);

								update_mapping(loc_nodes);
							}
							else if (loc_nodes.size() == 6)
							{
								tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[5]);
								tris.emplace_back(loc_nodes[3], loc_nodes[1], loc_nodes[4]);
								tris.emplace_back(loc_nodes[4], loc_nodes[2], loc_nodes[5]);
								tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[5]);

								update_mapping(loc_nodes);
							}
							else if (loc_nodes.size() == 10)
							{
								tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[8]);
								tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[9]);
								tris.emplace_back(loc_nodes[4], loc_nodes[1], loc_nodes[5]);
								tris.emplace_back(loc_nodes[5], loc_nodes[6], loc_nodes[9]);
								tris.emplace_back(loc_nodes[6], loc_nodes[2], loc_nodes[7]);
								tris.emplace_back(loc_nodes[7], loc_nodes[8], loc_nodes[9]);
								tris.emplace_back(loc_nodes[8], loc_nodes[3], loc_nodes[9]);
								tris.emplace_back(loc_nodes[9], loc_nodes[4], loc_nodes[5]);
								tris.emplace_back(loc_nodes[6], loc_nodes[7], loc_nodes[9]);
								update_mapping(loc_nodes);
							}
							else
							{
								logger().trace("skipping element {} since it is not linear, it has {} nodes", eid, loc_nodes.size());
							}
						}
						else
						{
							if (loc_nodes.size() < 4 || loc_local_nodes.size() < 4)
							{
								logger().trace("skipping prism quad face {} since it has only {} complete nodes", eid, loc_nodes.size());
								continue;
							}

							const int p = b.bases.empty() ? -1 : b.bases.front().order();
							const int n_tri_nodes = (p + 1) * (p + 2) / 2;
							const int q = n_tri_nodes > 0 && b.bases.size() % n_tri_nodes == 0 ? int(b.bases.size()) / n_tri_nodes - 1 : -1;

							if (p < 1 || p > 3 || q < 1 || q > 3 || (p == 3 && q == 3))
							{
								logger().trace("skipping prism quad face {} with unsupported p={}, q={}", eid, p, q);
								continue;
							}

							auto is_vertical_prism_edge = [](const int a, const int b) {
								return (a >= 0 && a < 3 && b == a + 3) || (b >= 0 && b < 3 && a == b + 3);
							};

							std::vector<int> edge_orders(4);
							for (int k = 0; k < 4; ++k)
								edge_orders[k] = is_vertical_prism_edge(loc_local_nodes[k], loc_local_nodes[(k + 1) % 4]) ? q : p;

							const int u_order = edge_orders[0];
							const int v_order = edge_orders[1];
							const int expected_nodes = (u_order + 1) * (v_order + 1);
							if (loc_nodes.size() != expected_nodes || edge_orders[0] != edge_orders[2] || edge_orders[1] != edge_orders[3])
							{
								logger().trace("skipping prism quad face {} with p={}, q={} and {} nodes", eid, p, q, loc_nodes.size());
								continue;
							}

							std::vector<int> grid(expected_nodes, -1);
							auto grid_index = [u_order](const int i, const int j) {
								return j * (u_order + 1) + i;
							};

							grid[grid_index(0, 0)] = loc_nodes[0];
							grid[grid_index(u_order, 0)] = loc_nodes[1];
							grid[grid_index(u_order, v_order)] = loc_nodes[2];
							grid[grid_index(0, v_order)] = loc_nodes[3];

							int node_index = 4;
							for (int i = 1; i < u_order; ++i)
								grid[grid_index(i, 0)] = loc_nodes[node_index++];
							for (int j = 1; j < v_order; ++j)
								grid[grid_index(u_order, j)] = loc_nodes[node_index++];
							for (int i = u_order - 1; i > 0; --i)
								grid[grid_index(i, v_order)] = loc_nodes[node_index++];
							for (int j = v_order - 1; j > 0; --j)
								grid[grid_index(0, j)] = loc_nodes[node_index++];

							for (int j = 1; j < v_order; ++j)
								for (int i = 1; i < u_order; ++i)
									grid[grid_index(i, j)] = loc_nodes[node_index++];

							assert(node_index == loc_nodes.size());
							assert(std::all_of(grid.begin(), grid.end(), [](const int n) { return n >= 0; }));

							for (int j = 0; j < v_order; ++j)
							{
								for (int i = 0; i < u_order; ++i)
								{
									tris.emplace_back(grid[grid_index(i, j)], grid[grid_index(i + 1, j)], grid[grid_index(i, j + 1)]);
									tris.emplace_back(grid[grid_index(i + 1, j + 1)], grid[grid_index(i, j + 1)], grid[grid_index(i + 1, j)]);
								}
							}

							update_mapping(loc_nodes);
						}

						continue;
					}
					else if (mesh.is_pyramid(lb.element_id()))
					{
						assert(!is_simplicial);
						assert(!mesh.has_poly());
						std::vector<int> loc_nodes;
						std::vector<int> loc_local_nodes;

						for (long n = 0; n < nodes.size(); ++n)
						{
							auto &bs = b.bases[nodes(n)];
							const auto &glob = bs.global();
							if (glob.size() != 1)
								continue;

							int gindex = glob.front().index;
							node_positions_vec.resize(std::max(int(node_positions_vec.size()), gindex + 1));
							node_positions_vec[gindex] = glob.front().node;
							loc_nodes.push_back(gindex);
							loc_local_nodes.push_back(nodes(n));
						}

						auto update_mapping = [&displacement_map_entries, &visited_node](const std::vector<int> &loc_nodes) {
							for (int k = 0; k < loc_nodes.size(); ++k)
							{
								if (!visited_node[loc_nodes[k]])
									displacement_map_entries.emplace_back(loc_nodes[k], loc_nodes[k], 1);

								visited_node[loc_nodes[k]] = true;
							}
						};

						const int p = b.bases.empty() ? -1 : b.bases.front().order();
						if (p < 1 || p > 3)
						{
							logger().trace("skipping pyramid face {} with unsupported p={}", eid, p);
							continue;
						}

						if (lid == 0)
						{
							const int expected_nodes = (p + 1) * (p + 1);
							if (loc_nodes.size() != expected_nodes || loc_local_nodes.size() != expected_nodes)
							{
								logger().trace("skipping pyramid quad face {} with p={} and {} nodes", eid, p, loc_nodes.size());
								continue;
							}

							Eigen::MatrixXd pyramid_nodes;
							autogen::pyramid_nodes_3d(p, pyramid_nodes);

							const Eigen::RowVector3d origin = pyramid_nodes.row(loc_local_nodes[0]);
							const Eigen::RowVector3d u_axis = pyramid_nodes.row(loc_local_nodes[1]) - origin;
							const Eigen::RowVector3d v_axis = pyramid_nodes.row(loc_local_nodes[3]) - origin;

							std::vector<int> grid(expected_nodes, -1);
							auto grid_index = [p](const int i, const int j) {
								return j * (p + 1) + i;
							};

							bool valid_grid = true;
							for (int n = 0; n < loc_nodes.size(); ++n)
							{
								const Eigen::RowVector3d rel = pyramid_nodes.row(loc_local_nodes[n]) - origin;
								const int i = int(std::lround(p * rel.dot(u_axis) / u_axis.squaredNorm()));
								const int j = int(std::lround(p * rel.dot(v_axis) / v_axis.squaredNorm()));
								if (i < 0 || i > p || j < 0 || j > p)
								{
									logger().trace("skipping pyramid quad face {} with invalid local grid coordinate ({}, {})", eid, i, j);
									valid_grid = false;
									break;
								}
								if (grid[grid_index(i, j)] >= 0)
								{
									logger().trace("skipping pyramid quad face {} with duplicate local grid coordinate ({}, {})", eid, i, j);
									valid_grid = false;
									break;
								}
								grid[grid_index(i, j)] = loc_nodes[n];
							}

							if (!valid_grid || !std::all_of(grid.begin(), grid.end(), [](const int n) { return n >= 0; }))
								continue;

							for (int j = 0; j < p; ++j)
							{
								for (int i = 0; i < p; ++i)
								{
									tris.emplace_back(grid[grid_index(i, j)], grid[grid_index(i + 1, j)], grid[grid_index(i, j + 1)]);
									tris.emplace_back(grid[grid_index(i + 1, j + 1)], grid[grid_index(i, j + 1)], grid[grid_index(i + 1, j)]);
								}
							}

							update_mapping(loc_nodes);
						}
						else if (loc_nodes.size() == 3)
						{
							tris.emplace_back(loc_nodes[0], loc_nodes[1], loc_nodes[2]);
							update_mapping(loc_nodes);
						}
						else if (loc_nodes.size() == 6)
						{
							tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[5]);
							tris.emplace_back(loc_nodes[3], loc_nodes[1], loc_nodes[4]);
							tris.emplace_back(loc_nodes[4], loc_nodes[2], loc_nodes[5]);
							tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[5]);
							update_mapping(loc_nodes);
						}
						else if (loc_nodes.size() == 10)
						{
							tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[8]);
							tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[9]);
							tris.emplace_back(loc_nodes[4], loc_nodes[1], loc_nodes[5]);
							tris.emplace_back(loc_nodes[5], loc_nodes[6], loc_nodes[9]);
							tris.emplace_back(loc_nodes[6], loc_nodes[2], loc_nodes[7]);
							tris.emplace_back(loc_nodes[7], loc_nodes[8], loc_nodes[9]);
							tris.emplace_back(loc_nodes[8], loc_nodes[3], loc_nodes[9]);
							tris.emplace_back(loc_nodes[9], loc_nodes[4], loc_nodes[5]);
							tris.emplace_back(loc_nodes[6], loc_nodes[7], loc_nodes[9]);
							update_mapping(loc_nodes);
						}
						else
						{
							logger().trace("skipping pyramid tri face {} with p={} and {} nodes", eid, p, loc_nodes.size());
							continue;
						}

						continue;
					}

					if (!mesh.is_simplex(lb.element_id()))
					{
						logger().trace("skipping element {} since it is not a simplex or hex", eid);
						continue;
					}

					assert(mesh.is_simplex(lb.element_id()));

					std::vector<int> loc_nodes;

					bool is_follower = false;
					if (!mesh3d.is_conforming())
					{
						for (long n = 0; n < nodes.size(); ++n)
						{
							auto &bs = b.bases[nodes(n)];
							const auto &glob = bs.global();
							if (glob.size() != 1)
							{
								is_follower = true;
								break;
							}
						}
					}

					if (is_follower)
						continue;

					for (long n = 0; n < nodes.size(); ++n)
					{
						const basis::Basis &bs = b.bases[nodes(n)];
						const std::vector<basis::Local2Global> &glob = bs.global();
						if (glob.size() != 1)
							continue;

						int gindex = glob.front().index;
						node_positions_vec.resize(std::max(int(node_positions_vec.size()), gindex + 1));
						node_positions_vec[gindex] = glob.front().node;
						loc_nodes.push_back(gindex);
					}

					if (loc_nodes.size() == 3)
					{
						tris.emplace_back(loc_nodes[0], loc_nodes[1], loc_nodes[2]);
					}
					else if (loc_nodes.size() == 6)
					{
						tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[5]);
						tris.emplace_back(loc_nodes[3], loc_nodes[1], loc_nodes[4]);
						tris.emplace_back(loc_nodes[4], loc_nodes[2], loc_nodes[5]);
						tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[5]);
					}
					else if (loc_nodes.size() == 10)
					{
						tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[8]);
						tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[9]);
						tris.emplace_back(loc_nodes[4], loc_nodes[1], loc_nodes[5]);
						tris.emplace_back(loc_nodes[5], loc_nodes[6], loc_nodes[9]);
						tris.emplace_back(loc_nodes[6], loc_nodes[2], loc_nodes[7]);
						tris.emplace_back(loc_nodes[7], loc_nodes[8], loc_nodes[9]);
						tris.emplace_back(loc_nodes[8], loc_nodes[3], loc_nodes[9]);
						tris.emplace_back(loc_nodes[9], loc_nodes[4], loc_nodes[5]);
						tris.emplace_back(loc_nodes[6], loc_nodes[7], loc_nodes[9]);
					}
					else if (loc_nodes.size() == 15)
					{
						tris.emplace_back(loc_nodes[0], loc_nodes[3], loc_nodes[11]);
						tris.emplace_back(loc_nodes[3], loc_nodes[4], loc_nodes[12]);
						tris.emplace_back(loc_nodes[3], loc_nodes[12], loc_nodes[11]);
						tris.emplace_back(loc_nodes[12], loc_nodes[10], loc_nodes[11]);
						tris.emplace_back(loc_nodes[4], loc_nodes[5], loc_nodes[13]);
						tris.emplace_back(loc_nodes[4], loc_nodes[13], loc_nodes[12]);
						tris.emplace_back(loc_nodes[12], loc_nodes[13], loc_nodes[14]);
						tris.emplace_back(loc_nodes[12], loc_nodes[14], loc_nodes[10]);
						tris.emplace_back(loc_nodes[14], loc_nodes[9], loc_nodes[10]);
						tris.emplace_back(loc_nodes[5], loc_nodes[1], loc_nodes[6]);
						tris.emplace_back(loc_nodes[5], loc_nodes[6], loc_nodes[13]);
						tris.emplace_back(loc_nodes[6], loc_nodes[7], loc_nodes[13]);
						tris.emplace_back(loc_nodes[13], loc_nodes[7], loc_nodes[14]);
						tris.emplace_back(loc_nodes[7], loc_nodes[8], loc_nodes[14]);
						tris.emplace_back(loc_nodes[14], loc_nodes[8], loc_nodes[9]);
						tris.emplace_back(loc_nodes[8], loc_nodes[2], loc_nodes[9]);
					}
					else
					{
						print_warning << loc_nodes.size() << " ";
						// assert(false);
					}

					if (!is_simplicial)
					{
						for (int k = 0; k < loc_nodes.size(); ++k)
						{
							if (!visited_node[loc_nodes[k]])
								displacement_map_entries.emplace_back(loc_nodes[k], loc_nodes[k], 1);

							visited_node[loc_nodes[k]] = true;
						}
					}
				}
			}

			if (print_warning.str().size() > 0)
				logger().warn("Skipping faces as theys have {} nodes, boundary export supported up to p4", print_warning.str());

			// downstream consumers (e.g. the shape-derivative code) expect every
			// FE node to have a row, as the pre-split extraction guaranteed
			node_positions_vec.resize(
				std::max(node_positions_vec.size(), size_t(n_bases + (is_simplicial ? 0 : mesh.n_faces()))),
				Eigen::Vector3d::Zero());

			node_positions.resize(node_positions_vec.size(), 3);
			for (int i = 0; i < node_positions_vec.size(); ++i)
				node_positions.row(i) = node_positions_vec[i];

			boundary_triangles.resize(tris.size(), 3);
			for (int i = 0; i < tris.size(); ++i)
			{
				boundary_triangles.row(i) << std::get<0>(tris[i]), std::get<2>(tris[i]), std::get<1>(tris[i]);
			}

			if (boundary_triangles.rows() > 0)
			{
				igl::edges(boundary_triangles, boundary_edges);
			}

			if (const char *dump = getenv("POLYFEM_DUMP_COLLISION_PROXY"))
				igl::write_triangle_mesh(dump, node_positions, boundary_triangles);
		}
		else
		{
			node_positions.resize(n_bases, 2);
			node_positions.setZero();
			const Mesh2D &mesh2d = dynamic_cast<const Mesh2D &>(mesh);

			std::vector<std::pair<int, int>> edges;

			for (const LocalBoundary &lb : total_local_boundary)
			{
				const basis::ElementBases &b = bases[lb.element_id()];

				for (int j = 0; j < lb.size(); ++j)
				{
					const int eid = lb.global_primitive_id(j);
					const int lid = lb[j];
					const Eigen::VectorXi nodes = b.local_nodes_for_primitive(eid, mesh2d);

					int prev_node = -1;

					for (long n = 0; n < nodes.size(); ++n)
					{
						const basis::Basis &bs = b.bases[nodes(n)];
						const std::vector<basis::Local2Global> &glob = bs.global();
						if (glob.size() != 1)
							continue;

						int gindex = glob.front().index;
						node_positions.row(gindex) = glob.front().node.head<2>();

						if (prev_node >= 0)
							edges.emplace_back(prev_node, gindex);

						prev_node = gindex;
					}
				}
			}

			boundary_triangles.resize(0, 0);
			boundary_edges.resize(edges.size(), 2);
			for (int i = 0; i < edges.size(); ++i)
			{
				boundary_edges.row(i) << edges[i].first, edges[i].second;
			}
		}
	}

	void OutGeometryData::build_vis_boundary_mesh(
		const mesh::Mesh &mesh,
		const std::vector<basis::ElementBases> &gbases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		Eigen::MatrixXd &boundary_vis_vertices,
		Eigen::MatrixXd &boundary_vis_local_vertices,
		Eigen::MatrixXi &boundary_vis_elements,
		Eigen::MatrixXi &boundary_vis_elements_ids,
		Eigen::MatrixXi &boundary_vis_primitive_ids,
		Eigen::MatrixXd &boundary_vis_normals) const
	{
		using namespace polyfem::mesh;

		std::vector<Eigen::MatrixXd> lv, vertices, allnormals;
		std::vector<int> el_ids, global_primitive_ids;
		Eigen::MatrixXd uv, local_pts, tmp_n, normals;
		assembler::ElementAssemblyValues vals;
		const auto &sampler = ref_element_sampler;
		const int n_samples = sampler.num_samples();
		int size = 0;

		std::vector<std::pair<int, int>> edges;
		std::vector<std::tuple<int, int, int>> tris;

		for (auto it = total_local_boundary.begin(); it != total_local_boundary.end(); ++it)
		{
			const auto &lb = *it;
			const auto &gbs = gbases[lb.element_id()];

			for (int k = 0; k < lb.size(); ++k)
			{
				switch (lb.type())
				{
				case BoundaryType::TRI_LINE:
					utils::BoundarySampler::normal_for_tri_edge(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_tri_edge(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::QUAD_LINE:
					utils::BoundarySampler::normal_for_quad_edge(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_quad_edge(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::QUAD:
					utils::BoundarySampler::normal_for_quad_face(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_quad_face(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::TRI:
					utils::BoundarySampler::normal_for_tri_face(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_tri_face(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::PRISM:
					utils::BoundarySampler::normal_for_prism_face(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_prism_face(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::PYRAMID:
					utils::BoundarySampler::normal_for_pyramid_face(lb[k], tmp_n);
					utils::BoundarySampler::sample_parametric_pyramid_face(lb[k], n_samples, uv, local_pts);
					break;
				case BoundaryType::POLYGON:
					utils::BoundarySampler::normal_for_polygon_edge(lb.element_id(), lb.global_primitive_id(k), mesh, tmp_n);
					utils::BoundarySampler::sample_polygon_edge(lb.element_id(), lb.global_primitive_id(k), n_samples, mesh, uv, local_pts);
					break;
				case BoundaryType::POLYHEDRON:
					assert(false);
					break;
				case BoundaryType::INVALID:
					assert(false);
					break;
				default:
					assert(false);
				}

				vertices.emplace_back();
				lv.emplace_back(local_pts);
				el_ids.push_back(lb.element_id());
				global_primitive_ids.push_back(lb.global_primitive_id(k));
				gbs.eval_geom_mapping(local_pts, vertices.back());
				vals.compute(lb.element_id(), mesh.is_volume(), local_pts, gbs, gbs);
				const int tris_start = tris.size();

				if (mesh.is_volume())
				{
					const bool prism_quad = lb.type() == BoundaryType::PRISM && lb[k] >= 2;
					const bool prism_tri = lb.type() == BoundaryType::PRISM && lb[k] < 2;

					const bool pyramid_quad = lb.type() == BoundaryType::PYRAMID && lb[k] == 0;
					const bool pyramid_tri = lb.type() == BoundaryType::PYRAMID && lb[k] > 0;

					if (lb.type() == BoundaryType::QUAD || prism_quad || pyramid_quad)
					{
						const auto map = [n_samples, size](int i, int j) { return j * n_samples + i + size; };

						for (int j = 0; j < n_samples - 1; ++j)
						{
							for (int i = 0; i < n_samples - 1; ++i)
							{
								tris.emplace_back(map(i, j), map(i + 1, j), map(i, j + 1));
								tris.emplace_back(map(i + 1, j + 1), map(i, j + 1), map(i + 1, j));
							}
						}
					}
					else if (lb.type() == BoundaryType::TRI || prism_tri || pyramid_tri)
					{
						int index = 0;
						std::vector<int> mapp(n_samples * n_samples, -1);
						for (int j = 0; j < n_samples; ++j)
						{
							for (int i = 0; i < n_samples - j; ++i)
							{
								mapp[j * n_samples + i] = index;
								++index;
							}
						}
						const auto map = [mapp, n_samples](int i, int j) {
							if (j * n_samples + i >= mapp.size())
								return -1;
							return mapp[j * n_samples + i];
						};

						for (int j = 0; j < n_samples - 1; ++j)
						{
							for (int i = 0; i < n_samples - j; ++i)
							{
								if (map(i, j) >= 0 && map(i + 1, j) >= 0 && map(i, j + 1) >= 0)
									tris.emplace_back(map(i, j) + size, map(i + 1, j) + size, map(i, j + 1) + size);

								if (map(i + 1, j + 1) >= 0 && map(i, j + 1) >= 0 && map(i + 1, j) >= 0)
									tris.emplace_back(map(i + 1, j + 1) + size, map(i, j + 1) + size, map(i + 1, j) + size);
							}
						}
					}
					else
					{
						assert(false);
					}
				}
				else
				{
					for (int i = 0; i < vertices.back().rows() - 1; ++i)
						edges.emplace_back(i + size, i + size + 1);
				}

				normals.resize(vals.jac_it.size(), tmp_n.cols());

				for (int n = 0; n < vals.jac_it.size(); ++n)
				{
					normals.row(n) = tmp_n * vals.jac_it[n];
					normals.row(n).normalize();
				}

				allnormals.push_back(normals);

				tmp_n.setZero();
				for (int n = 0; n < vals.jac_it.size(); ++n)
				{
					tmp_n += normals.row(n);
				}

				if (mesh.is_volume())
				{
					Eigen::Vector3d e1 = vertices.back().row(std::get<1>(tris.back()) - size) - vertices.back().row(std::get<0>(tris.back()) - size);
					Eigen::Vector3d e2 = vertices.back().row(std::get<2>(tris.back()) - size) - vertices.back().row(std::get<0>(tris.back()) - size);

					Eigen::Vector3d n = e1.cross(e2);
					Eigen::Vector3d nn = tmp_n.transpose();

					if (n.dot(nn) < 0)
					{
						for (int i = tris_start; i < tris.size(); ++i)
						{
							tris[i] = std::tuple<int, int, int>(std::get<0>(tris[i]), std::get<2>(tris[i]), std::get<1>(tris[i]));
						}
					}
				}

				size += vertices.back().rows();
			}
		}

		boundary_vis_vertices.resize(size, vertices.front().cols());
		boundary_vis_local_vertices.resize(size, vertices.front().cols());
		boundary_vis_elements_ids.resize(size, 1);
		boundary_vis_primitive_ids.resize(size, 1);
		boundary_vis_normals.resize(size, vertices.front().cols());

		if (mesh.is_volume())
			boundary_vis_elements.resize(tris.size(), 3);
		else
			boundary_vis_elements.resize(edges.size(), 2);

		int index = 0;
		int ii = 0;
		for (const auto &v : vertices)
		{
			boundary_vis_vertices.block(index, 0, v.rows(), v.cols()) = v;
			boundary_vis_local_vertices.block(index, 0, v.rows(), v.cols()) = lv[ii];
			boundary_vis_elements_ids.block(index, 0, v.rows(), 1).setConstant(el_ids[ii]);
			boundary_vis_primitive_ids.block(index, 0, v.rows(), 1).setConstant(global_primitive_ids[ii++]);
			index += v.rows();
		}

		index = 0;
		for (const auto &n : allnormals)
		{
			boundary_vis_normals.block(index, 0, n.rows(), n.cols()) = n;
			index += n.rows();
		}

		index = 0;
		if (mesh.is_volume())
		{
			for (const auto &t : tris)
			{
				boundary_vis_elements.row(index) << std::get<0>(t), std::get<1>(t), std::get<2>(t);
				++index;
			}
		}
		else
		{
			for (const auto &e : edges)
			{
				boundary_vis_elements.row(index) << e.first, e.second;
				++index;
			}
		}
	}

	void OutGeometryData::build_vis_mesh(
		const mesh::Mesh &mesh,
		const Eigen::VectorXi &disc_orders,
		const std::vector<basis::ElementBases> &gbases,
		const std::map<int, Eigen::MatrixXd> &polys,
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d,
		const bool boundary_only,
		Eigen::MatrixXd &points,
		Eigen::MatrixXi &tets,
		Eigen::MatrixXi &el_id,
		Eigen::MatrixXd &discr,
		Eigen::MatrixXd &local_points) const
	{
		const auto &sampler = ref_element_sampler;

		const auto &current_bases = gbases;
		int tet_total_size = 0;
		int pts_total_size = 0;

		Eigen::MatrixXd vis_pts_poly;
		Eigen::MatrixXi vis_faces_poly, vis_edges_poly;

		for (size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if (boundary_only && mesh.is_volume() && !mesh.is_boundary_element(i))
				continue;

			if (mesh.is_simplex(i))
			{
				tet_total_size += sampler.simplex_volume().rows();
				pts_total_size += sampler.simplex_points().rows();
			}
			else if (mesh.is_cube(i))
			{
				tet_total_size += sampler.cube_volume().rows();
				pts_total_size += sampler.cube_points().rows();
			}
			else if (mesh.is_prism(i))
			{
				tet_total_size += sampler.prism_volume().rows();
				pts_total_size += sampler.prism_points().rows();
			}
			else if (mesh.is_pyramid(i))
			{
				tet_total_size += sampler.pyramid_volume().rows();
				pts_total_size += sampler.pyramid_points().rows();
			}
			else
			{
				if (mesh.is_volume())
				{
					sampler.sample_polyhedron(polys_3d.at(i).first, polys_3d.at(i).second, vis_pts_poly, vis_faces_poly, vis_edges_poly);

					tet_total_size += vis_faces_poly.rows();
					pts_total_size += vis_pts_poly.rows();
				}
				else
				{
					sampler.sample_polygon(polys.at(i), vis_pts_poly, vis_faces_poly, vis_edges_poly);

					tet_total_size += vis_faces_poly.rows();
					pts_total_size += vis_pts_poly.rows();
				}
			}
		}

		points.resize(pts_total_size, mesh.dimension());
		local_points.resize(pts_total_size, mesh.dimension());
		local_points.setZero();
		tets.resize(tet_total_size, mesh.is_volume() ? 4 : 3);

		el_id.resize(pts_total_size, 1);
		discr.resize(pts_total_size, 1);

		Eigen::MatrixXd mapped, tmp;
		int tet_index = 0, pts_index = 0;

		for (size_t i = 0; i < current_bases.size(); ++i)
		{
			const auto &bs = current_bases[i];

			if (boundary_only && mesh.is_volume() && !mesh.is_boundary_element(i))
				continue;

			if (mesh.is_simplex(i))
			{
				bs.eval_geom_mapping(sampler.simplex_points(), mapped);

				tets.block(tet_index, 0, sampler.simplex_volume().rows(), tets.cols()) = sampler.simplex_volume().array() + pts_index;
				tet_index += sampler.simplex_volume().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				local_points.block(pts_index, 0, sampler.simplex_points().rows(), sampler.simplex_points().cols()) = sampler.simplex_points();
				discr.block(pts_index, 0, mapped.rows(), 1).setConstant(disc_orders(i));
				el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
				pts_index += mapped.rows();
			}
			else if (mesh.is_cube(i))
			{
				bs.eval_geom_mapping(sampler.cube_points(), mapped);

				tets.block(tet_index, 0, sampler.cube_volume().rows(), tets.cols()) = sampler.cube_volume().array() + pts_index;
				tet_index += sampler.cube_volume().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				local_points.block(pts_index, 0, sampler.cube_points().rows(), sampler.cube_points().cols()) = sampler.cube_points();
				discr.block(pts_index, 0, mapped.rows(), 1).setConstant(disc_orders(i));
				el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
				pts_index += mapped.rows();
			}
			else if (mesh.is_prism(i))
			{
				bs.eval_geom_mapping(sampler.prism_points(), mapped);

				tets.block(tet_index, 0, sampler.prism_volume().rows(), tets.cols()) = sampler.prism_volume().array() + pts_index;
				tet_index += sampler.prism_volume().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				local_points.block(pts_index, 0, sampler.prism_points().rows(), sampler.prism_points().cols()) = sampler.prism_points();
				discr.block(pts_index, 0, mapped.rows(), 1).setConstant(disc_orders(i));
				el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
				pts_index += mapped.rows();
			}
			else if (mesh.is_pyramid(i))
			{
				bs.eval_geom_mapping(sampler.pyramid_points(), mapped);

				tets.block(tet_index, 0, sampler.pyramid_volume().rows(), tets.cols()) = sampler.pyramid_volume().array() + pts_index;
				tet_index += sampler.pyramid_volume().rows();

				points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
				local_points.block(pts_index, 0, sampler.pyramid_points().rows(), sampler.pyramid_points().cols()) = sampler.pyramid_points();
				discr.block(pts_index, 0, mapped.rows(), 1).setConstant(disc_orders(i));
				el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
				pts_index += mapped.rows();
			}
			else
			{
				if (mesh.is_volume())
				{
					sampler.sample_polyhedron(polys_3d.at(i).first, polys_3d.at(i).second, vis_pts_poly, vis_faces_poly, vis_edges_poly);
					bs.eval_geom_mapping(vis_pts_poly, mapped);

					tets.block(tet_index, 0, vis_faces_poly.rows(), tets.cols()) = vis_faces_poly.array() + pts_index;
					tet_index += vis_faces_poly.rows();

					points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
					local_points.block(pts_index, 0, vis_pts_poly.rows(), vis_pts_poly.cols()) = vis_pts_poly;
					discr.block(pts_index, 0, mapped.rows(), 1).setConstant(-1);
					el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
					pts_index += mapped.rows();
				}
				else
				{
					sampler.sample_polygon(polys.at(i), vis_pts_poly, vis_faces_poly, vis_edges_poly);
					bs.eval_geom_mapping(vis_pts_poly, mapped);

					tets.block(tet_index, 0, vis_faces_poly.rows(), tets.cols()) = vis_faces_poly.array() + pts_index;
					tet_index += vis_faces_poly.rows();

					points.block(pts_index, 0, mapped.rows(), points.cols()) = mapped;
					local_points.block(pts_index, 0, vis_pts_poly.rows(), vis_pts_poly.cols()) = vis_pts_poly;
					discr.block(pts_index, 0, mapped.rows(), 1).setConstant(-1);
					el_id.block(pts_index, 0, mapped.rows(), 1).setConstant(i);
					pts_index += mapped.rows();
				}
			}
		}

		assert(pts_index == points.rows());
		assert(tet_index == tets.rows());
	}

	void OutGeometryData::build_high_order_vis_mesh(
		const mesh::Mesh &mesh,
		const Eigen::VectorXi &output_orders,
		const std::vector<basis::ElementBases> &bases,
		Eigen::MatrixXd &points,
		std::vector<CellElement> &elements,
		Eigen::MatrixXi &el_id,
		Eigen::MatrixXd &discr,
		Eigen::MatrixXd &local_points) const
	{
		// if (!mesh)
		// {
		// 	logger().error("Load the mesh first!");
		// 	return;
		// }
		// if (n_bases <= 0)
		// {
		// 	logger().error("Build the bases first!");
		// 	return;
		// }
		// assert(mesh.is_linear());

		std::vector<RowVectorNd> nodes;
		int pts_total_size = 0;
		elements.resize(bases.size());
		Eigen::MatrixXd ref_pts;

		for (size_t i = 0; i < bases.size(); ++i)
		{
			const auto &bs = bases[i];
			if (mesh.is_volume())
			{
				if (mesh.is_simplex(i))
					autogen::p_nodes_3d(output_orders(i), ref_pts);
				else if (mesh.is_cube(i))
					autogen::q_nodes_3d(output_orders(i), ref_pts);
				else if (mesh.is_prism(i))
				{
					autogen::prism_nodes_3d(output_orders(i), output_orders(i), ref_pts);
				}
				else if (mesh.is_pyramid(i))
				{
					if (output_orders(i) == 1)
						pyramid_nodes_for_output(1, ref_pts);
					else
						ref_pts = ref_element_sampler.pyramid_points();
				}
				else
					continue;
			}
			else
			{
				if (mesh.is_simplex(i))
					autogen::p_nodes_2d(output_orders(i), ref_pts);
				else if (mesh.is_cube(i))
					autogen::q_nodes_2d(output_orders(i), ref_pts);
				else
				{
					const int n_v = static_cast<const mesh::Mesh2D &>(mesh).n_face_vertices(i);
					ref_pts.resize(n_v, 2);
				}
			}

			pts_total_size += ref_pts.rows();
		}

		points.resize(pts_total_size, mesh.dimension());
		local_points.resize(pts_total_size, mesh.dimension());
		local_points.setZero();

		el_id.resize(pts_total_size, 1);
		discr.resize(pts_total_size, 1);

		Eigen::MatrixXd mapped;
		int pts_index = 0;

		std::string error_msg = "";

		for (size_t i = 0; i < bases.size(); ++i)
		{
			const auto &bs = bases[i];
			if (mesh.is_volume())
			{
				if (mesh.is_simplex(i))
					autogen::p_nodes_3d(output_orders(i), ref_pts);
				else if (mesh.is_cube(i))
					autogen::q_nodes_3d(output_orders(i), ref_pts);
				else if (mesh.is_prism(i))
				{
					autogen::prism_nodes_3d(output_orders(i), output_orders(i), ref_pts);
				}
				else if (mesh.is_pyramid(i))
				{
					if (output_orders(i) == 1)
						pyramid_nodes_for_output(1, ref_pts);
					else
						ref_pts = ref_element_sampler.pyramid_points();
				}
				else
					continue;
			}
			else
			{
				if (mesh.is_simplex(i))
					autogen::p_nodes_2d(output_orders(i), ref_pts);
				else if (mesh.is_cube(i))
					autogen::q_nodes_2d(output_orders(i), ref_pts);
				else
					continue;
			}

			bs.eval_geom_mapping(ref_pts, mapped);

			for (int j = 0; j < mapped.rows(); ++j)
			{
				points.row(pts_index) = mapped.row(j);
				local_points.row(pts_index).leftCols(ref_pts.cols()) = ref_pts.row(j);
				el_id(pts_index) = i;
				discr(pts_index) = output_orders(i);
				elements[i].vertices.push_back(pts_index);

				pts_index++;
			}

			if (mesh.is_simplex(i))
			{
				if (mesh.is_volume())
				{
					const int n_nodes = elements[i].vertices.size();
					if (output_orders(i) >= 3)
					{
						std::swap(elements[i].vertices[16], elements[i].vertices[17]);
						std::swap(elements[i].vertices[17], elements[i].vertices[18]);
						std::swap(elements[i].vertices[18], elements[i].vertices[19]);
					}
					if (output_orders(i) > 4)
						error_msg = "Saving high-order meshes not implemented for P5+ elements!";
				}
				else
				{
					if (output_orders(i) == 4)
					{
						const int n_nodes = elements[i].vertices.size();
						std::swap(elements[i].vertices[n_nodes - 1], elements[i].vertices[n_nodes - 2]);
					}
					if (output_orders(i) > 4)
						error_msg = "Saving high-order meshes not implemented for P5+ elements!";
				}
			}
			else if (mesh.is_cube(i) && mesh.is_volume())
			{
				const int n_nodes = elements[i].vertices.size();
				if (output_orders(i) == 2) // Lagrange hex, order=2
				{
					std::swap(elements[i].vertices[12], elements[i].vertices[16]);
					std::swap(elements[i].vertices[13], elements[i].vertices[17]);
					std::swap(elements[i].vertices[14], elements[i].vertices[18]);
					std::swap(elements[i].vertices[15], elements[i].vertices[19]);
					std::swap(elements[i].vertices[18], elements[i].vertices[19]); // a hack fix
				}
				// if (disc_orders(i) == 3)  // Incomplete fix, need to fix order on the edge
				// {
				// 	std::swap(elements[i].vertices[24], elements[i].vertices[16]);
				// 	std::swap(elements[i].vertices[25], elements[i].vertices[17]);
				// 	std::swap(elements[i].vertices[26], elements[i].vertices[18]);
				// 	std::swap(elements[i].vertices[27], elements[i].vertices[19]);
				// 	std::swap(elements[i].vertices[28], elements[i].vertices[20]);
				// 	std::swap(elements[i].vertices[29], elements[i].vertices[21]);
				// 	std::swap(elements[i].vertices[30], elements[i].vertices[22]);
				// 	std::swap(elements[i].vertices[31], elements[i].vertices[23]);
				// 	std::swap(elements[i].vertices[28], elements[i].vertices[30]);  // hack
				// 	std::swap(elements[i].vertices[29], elements[i].vertices[31]);  // hack
				// }
				if (output_orders(i) > 2)
					error_msg = "Saving high-order meshes not implemented for Q2+ elements!";
			}
			else if (output_orders(i) > 1)
			{
				if (mesh.is_cube(i))
					error_msg = "Saving high-order meshes not implemented for Q2+ elements!";
			}
		}

		if (!error_msg.empty())
			logger().warn(error_msg);

		for (size_t i = 0; i < bases.size(); ++i)
		{
			if (mesh.is_volume() || !mesh.is_polytope(i))
				continue;

			const auto &mesh2d = static_cast<const mesh::Mesh2D &>(mesh);
			const int n_v = mesh2d.n_face_vertices(i);

			for (int j = 0; j < n_v; ++j)
			{
				points.row(pts_index) = mesh2d.point(mesh2d.face_vertex(i, j));
				local_points.row(pts_index) = mesh2d.point(mesh2d.face_vertex(i, j));
				el_id(pts_index) = i;
				discr(pts_index) = output_orders(i);
				elements[i].vertices.push_back(pts_index);

				pts_index++;
			}
		}

		for (size_t i = 0; i < bases.size(); ++i)
		{
			if (!mesh.is_volume())
			{
				if (elements[i].vertices.size() == 1)
					elements[i].ctype = CellType::Vertex;
				else if (elements[i].vertices.size() == 2)
					elements[i].ctype = CellType::Line;
				else if (mesh.is_simplex(i))
					elements[i].ctype = CellType::Triangle;
				else if (mesh.is_cube(i))
					elements[i].ctype = CellType::Quadrilateral;
				else
					elements[i].ctype = CellType::Polygon;
			}
			else
			{
				if (mesh.is_simplex(i))
					elements[i].ctype = CellType::Tetrahedron;
				else if (mesh.is_cube(i))
					elements[i].ctype = CellType::Hexahedron;
				else if (mesh.is_prism(i))
					elements[i].ctype = CellType::Wedge;
				else if (mesh.is_pyramid(i))
					elements[i].ctype = CellType::Pyramid;
			}
		}

		if (mesh.is_volume())
		{
			// ParaView does not reliably render high-order Lagrange pyramids;
			// tessellate only those cells while keeping linear pyramids and the
			// other element types in their native representation.
			std::vector<CellElement> expanded_elements;
			expanded_elements.reserve(elements.size());
			for (size_t i = 0; i < bases.size(); ++i)
			{
				if (!mesh.is_pyramid(i) || output_orders(i) == 1)
				{
					expanded_elements.push_back(std::move(elements[i]));
					continue;
				}

				for (int t = 0; t < ref_element_sampler.pyramid_volume().rows(); ++t)
				{
					CellElement tet;
					tet.ctype = CellType::Tetrahedron;
					for (int j = 0; j < ref_element_sampler.pyramid_volume().cols(); ++j)
						tet.vertices.push_back(elements[i].vertices[ref_element_sampler.pyramid_volume()(t, j)]);
					expanded_elements.push_back(std::move(tet));
				}
			}
			elements.swap(expanded_elements);
		}

		assert(pts_index == points.rows());
	}

	void OutGeometryData::export_data(
		const OutputSpace &space,
		const OutputFieldFunction &output_fields,
		const bool is_time_dependent,
		const double tend_in,
		const double dt,
		const ExportOptions &opts,
		const std::string &vis_mesh_path) const
	{
		if (!space.mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}

		double tend = tend_in;
		if (tend <= 0)
			tend = 1;

		if (!vis_mesh_path.empty() && !is_time_dependent)
		{
			save_vtu(
				vis_mesh_path, space, output_fields,
				tend, dt, opts);
		}
	}

	bool OutGeometryData::ExportOptions::export_field(const std::string &field) const
	{
		return fields.empty() || std::find(fields.begin(), fields.end(), field) != fields.end();
	}

	OutGeometryData::ExportOptions::ExportOptions(const json &args, const bool is_mesh_linear, const bool mesh_has_prisms, const bool is_problem_scalar)
	{
		fields = args["output"]["paraview"]["fields"];

		volume = args["output"]["paraview"]["volume"];
		surface = args["output"]["paraview"]["surface"];
		wire = args["output"]["paraview"]["wireframe"];
		points = args["output"]["paraview"]["points"];
		contact_forces = args["output"]["paraview"]["options"]["contact_forces"] && !is_problem_scalar;
		friction_forces = args["output"]["paraview"]["options"]["friction_forces"] && !is_problem_scalar;
		normal_adhesion_forces = args["output"]["paraview"]["options"]["normal_adhesion_forces"] && !is_problem_scalar;
		tangential_adhesion_forces = args["output"]["paraview"]["options"]["tangential_adhesion_forces"] && !is_problem_scalar;

		if (args["output"]["paraview"]["options"]["force_high_order"])
			use_sampler = false;
		else
			use_sampler = !(is_mesh_linear && args["output"]["paraview"]["high_order_mesh"]);
		boundary_only = use_sampler && args["output"]["advanced"]["vis_boundary_only"];
		sol_on_grid = args["output"]["advanced"]["sol_on_grid"] > 0;

		discretization_order = args["output"]["paraview"]["options"]["discretization_order"];

		reorder_output = args["output"]["data"]["advanced"]["reorder_nodes"];

		use_hdf5 = args["output"]["paraview"]["options"]["use_hdf5"];
	}

	void OutGeometryData::save_vtu(
		const std::string &path,
		const OutputSpace &space,
		const OutputFieldFunction &output_fields,
		const double t,
		const double dt,
		const ExportOptions &opts) const
	{
		if (!space.mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}

		const std::filesystem::path fs_path(path);
		const std::string path_stem = fs_path.stem().string();
		const std::string base_path = (fs_path.parent_path() / path_stem).string();
		paraviewo::VTMWriter vtm(t);
		save_vtu(path, space, output_fields, t, dt, opts, vtm, "");
		vtm.save(base_path + ".vtm");
	}

	void OutGeometryData::save_vtu(
		const std::string &path,
		const OutputSpace &space,
		const OutputFieldFunction &output_fields,
		const double t,
		const double dt,
		const ExportOptions &opts,
		paraviewo::VTMWriter &vtm,
		const std::string &block_prefix) const
	{
		if (!space.mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}

		const bool save_contact =
			space.collision_mesh
			&& (opts.contact_forces || opts.friction_forces || opts.normal_adhesion_forces || opts.tangential_adhesion_forces
				|| (!opts.fields.empty() && opts.export_field("adaptive_dhat")));

		logger().info("Saving vtu to {}; volume={}, surface={}, contact={}, points={}, wireframe={}",
					  path, opts.volume, opts.surface, save_contact, opts.points, opts.wire);

		const std::filesystem::path fs_path(path);
		const std::string path_stem = fs_path.stem().string();
		const std::string base_path = (fs_path.parent_path() / path_stem).string();

		if (opts.volume)
		{
			save_volume(base_path + opts.file_extension(), space, output_fields, t, dt, opts);
		}

		if (opts.surface)
		{
			save_surface(base_path + "_surf" + opts.file_extension(), space, output_fields, t, dt, opts);
		}

		if (save_contact)
		{
			save_contact_surface(base_path + "_surf" + opts.file_extension(), space, output_fields, t, dt, opts);
		}

		if (opts.wire)
		{
			save_wire(base_path + "_wire" + opts.file_extension(), space, output_fields, t, opts);
		}

		if (opts.points)
		{
			save_points(base_path + "_points" + opts.file_extension(), space, output_fields, opts);
		}

		const auto block_name = [&block_prefix](const std::string &name) {
			return block_prefix.empty() ? name : block_prefix + " " + name;
		};
		if (opts.volume)
			vtm.add_dataset(block_name("Volume"), "data", path_stem + opts.file_extension());
		if (opts.surface)
			vtm.add_dataset(block_name("Surface"), "data", path_stem + "_surf" + opts.file_extension());
		if (save_contact)
			vtm.add_dataset(block_name("Contact"), "data", path_stem + "_surf_contact" + opts.file_extension());
		if (opts.wire)
			vtm.add_dataset(block_name("Wireframe"), "data", path_stem + "_wire" + opts.file_extension());
		if (opts.points)
			vtm.add_dataset(block_name("Points"), "data", path_stem + "_points" + opts.file_extension());
	}

	void OutGeometryData::save_volume(
		const std::string &path,
		const OutputSpace &space,
		const OutputFieldFunction &output_fields,
		const double t,
		const double dt,
		const ExportOptions &opts) const
	{
		if (!space.mesh || !space.geometry_bases)
			return;

		static const std::map<int, Eigen::MatrixXd> empty_polys;
		static const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> empty_polys_3d;

		const mesh::Mesh &mesh = *space.mesh;
		const std::vector<basis::ElementBases> &gbases = *space.geometry_bases;
		const std::map<int, Eigen::MatrixXd> &polys = space.polys ? *space.polys : empty_polys;
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d = space.polys_3d ? *space.polys_3d : empty_polys_3d;
		const Eigen::VectorXi output_orders =
			space.output_orders.size() == mesh.n_elements()
				? space.output_orders
				: Eigen::VectorXi::Ones(mesh.n_elements());
		const mesh::Obstacle *obstacle = space.obstacle;

		Eigen::MatrixXd points;
		Eigen::MatrixXi tets;
		Eigen::MatrixXi el_id;
		Eigen::MatrixXd discr;
		Eigen::MatrixXd local_points;
		std::vector<CellElement> elements;

		if (opts.use_sampler)
			build_vis_mesh(mesh, output_orders, gbases,
						   polys, polys_3d, opts.boundary_only,
						   points, tets, el_id, discr, local_points);
		else
		{
			build_high_order_vis_mesh(mesh, output_orders, gbases,
									  points, elements, el_id, discr, local_points);
		}

		std::shared_ptr<paraviewo::ParaviewWriter> tmpw;
		if (opts.use_hdf5)
			tmpw = std::make_shared<paraviewo::HDF5VTUWriter>();
		else
			tmpw = std::make_shared<paraviewo::VTUWriter>();
		paraviewo::ParaviewWriter &writer = *tmpw;

		if (obstacle && obstacle->n_vertices() > 0)
		{
			discr.conservativeResize(discr.size() + obstacle->n_vertices(), 1);
			discr.bottomRows(obstacle->n_vertices()).setZero();
		}

		if (opts.discretization_order && opts.export_field("discr"))
			writer.add_field("discr", discr);

		if (obstacle && obstacle->n_vertices() > 0)
		{
			const int orig_p = points.rows();
			points.conservativeResize(points.rows() + obstacle->n_vertices(), points.cols());
			points.bottomRows(obstacle->n_vertices()) = obstacle->v();

			if (elements.empty())
			{
				for (int i = 0; i < tets.rows(); ++i)
				{
					elements.emplace_back();
					elements.back().ctype = mesh.is_volume() ? CellType::Tetrahedron : CellType::Triangle;
					for (int j = 0; j < tets.cols(); ++j)
						elements.back().vertices.push_back(tets(i, j));
				}
			}

			for (int i = 0; i < obstacle->get_face_connectivity().rows(); ++i)
			{
				elements.emplace_back();
				elements.back().ctype = CellType::Triangle;
				for (int j = 0; j < obstacle->get_face_connectivity().cols(); ++j)
					elements.back().vertices.push_back(obstacle->get_face_connectivity()(i, j) + orig_p);
			}

			for (int i = 0; i < obstacle->get_edge_connectivity().rows(); ++i)
			{
				elements.emplace_back();
				elements.back().ctype = CellType::Line;
				for (int j = 0; j < obstacle->get_edge_connectivity().cols(); ++j)
					elements.back().vertices.push_back(obstacle->get_edge_connectivity()(i, j) + orig_p);
			}

			for (int i = 0; i < obstacle->get_vertex_connectivity().size(); ++i)
			{
				elements.emplace_back();
				elements.back().ctype = CellType::Vertex;
				elements.back().vertices.push_back(obstacle->get_vertex_connectivity()(i) + orig_p);
			}
		}

		// Write the solution alias last so it is the default for warp-by-vector.
		OutputSample sample;
		sample.points = points;
		sample.local_points = local_points;
		sample.element_ids = el_id.col(0);
		sample.domain = OutputSample::Domain::Volume;
		sample.cell_count = elements.empty() ? tets.rows() : static_cast<int>(elements.size());
		sample.time = t;
		sample.dt = dt;
		add_output_fields(writer, sample, output_fields);

		if (opts.sol_on_grid && output_fields && grid_points.rows() > 0)
		{
			OutputSample grid_sample;
			grid_sample.points = grid_points;
			grid_sample.element_ids = grid_points_to_elements.col(0);
			grid_sample.domain = OutputSample::Domain::Grid;
			grid_sample.local_points.resize(grid_points.rows(), mesh.dimension());
			grid_sample.local_points.setZero();
			for (int i = 0; i < grid_points.rows(); ++i)
			{
				if (grid_sample.element_ids(i) >= 0)
					grid_sample.local_points.row(i) = grid_points_bc.row(i).rightCols(mesh.dimension());
			}
			grid_sample.time = t;
			grid_sample.dt = dt;
			grid_sample.requested_fields = {
				"solution",
				"solution_gradient",
				"pressure",
				"pressure_gradient",
			};

			io::write_matrix(path + "_grid.txt", grid_points);
			for (const OutputField &field : output_fields(grid_sample))
			{
				if (field.association != OutputField::Association::Point || field.values.rows() != grid_points.rows())
					continue;
				if (field.name == "solution")
					io::write_matrix(path + "_sol.txt", field.values);
				else if (field.name == "solution_gradient")
					io::write_matrix(path + "_grad.txt", field.values);
				else if (field.name == "pressure")
					io::write_matrix(path + "_p_sol.txt", field.values);
				else if (field.name == "pressure_gradient")
					io::write_matrix(path + "_p_grad.txt", field.values);
			}
		}

		if (elements.empty())
			writer.write_mesh(path, points, tets, mesh.is_volume() ? CellType::Tetrahedron : CellType::Triangle);
		else
			writer.write_mesh(path, points, elements);
	}

	void OutGeometryData::save_surface(
		const std::string &export_surface,
		const OutputSpace &space,
		const OutputFieldFunction &output_fields,
		const double t,
		const double dt_in,
		const ExportOptions &opts) const
	{
		if (!space.mesh || !space.geometry_bases || !space.total_local_boundary)
			return;

		const mesh::Mesh &mesh = *space.mesh;
		const std::vector<basis::ElementBases> &gbases = *space.geometry_bases;

		Eigen::MatrixXd boundary_vis_vertices;
		Eigen::MatrixXd boundary_vis_local_vertices;
		Eigen::MatrixXi boundary_vis_elements;
		Eigen::MatrixXi boundary_vis_elements_ids;
		Eigen::MatrixXi boundary_vis_primitive_ids;
		Eigen::MatrixXd boundary_vis_normals;

		build_vis_boundary_mesh(mesh, gbases, *space.total_local_boundary,
								boundary_vis_vertices, boundary_vis_local_vertices, boundary_vis_elements,
								boundary_vis_elements_ids, boundary_vis_primitive_ids, boundary_vis_normals);

		Eigen::MatrixXd discr, b_sidesets;
		discr.resize(boundary_vis_vertices.rows(), 1);
		b_sidesets.resize(boundary_vis_vertices.rows(), 1);
		b_sidesets.setZero();

		for (int i = 0; i < boundary_vis_vertices.rows(); ++i)
		{
			const auto s_id = mesh.get_boundary_id(boundary_vis_primitive_ids(i));
			if (s_id > 0)
			{
				b_sidesets(i) = s_id;
			}

			const int el_index = boundary_vis_elements_ids(i);
			discr(i) = space.output_orders.size() == mesh.n_elements() ? space.output_orders(el_index) : 1;
		}

		std::shared_ptr<paraviewo::ParaviewWriter> tmpw;
		if (opts.use_hdf5)
			tmpw = std::make_shared<paraviewo::HDF5VTUWriter>();
		else
			tmpw = std::make_shared<paraviewo::VTUWriter>();
		paraviewo::ParaviewWriter &writer = *tmpw;

		if (opts.export_field("normals"))
			writer.add_field("normals", boundary_vis_normals);
		if (opts.export_field("discr"))
			writer.add_field("discr", discr);
		if (opts.export_field("sidesets"))
			writer.add_field("sidesets", b_sidesets);

		// Write the solution alias last so it is the default for warp-by-vector.
		OutputSample sample;
		sample.points = boundary_vis_vertices;
		sample.local_points = boundary_vis_local_vertices;
		sample.element_ids = boundary_vis_elements_ids.col(0);
		sample.primitive_ids = boundary_vis_primitive_ids;
		sample.normals = boundary_vis_normals;
		sample.domain = OutputSample::Domain::Surface;
		sample.cell_count = boundary_vis_elements.rows();
		sample.time = t;
		sample.dt = dt_in;
		add_output_fields(writer, sample, output_fields);
		writer.write_mesh(export_surface, boundary_vis_vertices, boundary_vis_elements, mesh.is_volume() ? CellType::Triangle : CellType::Line);
	}

	void OutGeometryData::save_contact_surface(
		const std::string &export_surface,
		const OutputSpace &space,
		const OutputFieldFunction &output_fields,
		const double t,
		const double dt_in,
		const ExportOptions &opts) const
	{
		if (!space.collision_mesh)
			return;

		const ipc::CollisionMesh &collision_mesh = *space.collision_mesh;

		std::shared_ptr<paraviewo::ParaviewWriter> tmpw;
		if (opts.use_hdf5)
			tmpw = std::make_shared<paraviewo::HDF5VTUWriter>();
		else
			tmpw = std::make_shared<paraviewo::VTUWriter>();
		paraviewo::ParaviewWriter &writer = *tmpw;

		// Write the solution alias last so it is the default for warp-by-vector.
		OutputSample sample;
		sample.points = collision_mesh.rest_positions();
		sample.domain = OutputSample::Domain::Contact;
		sample.cell_count = static_cast<int>(
			collision_mesh.dim() == 3 ? collision_mesh.num_faces() : collision_mesh.num_edges());
		sample.time = t;
		sample.dt = dt_in;
		add_output_fields(writer, sample, output_fields);

		const std::filesystem::path surface_path(export_surface);
		const std::string contact_path =
			(surface_path.parent_path() / (surface_path.stem().string() + "_contact" + surface_path.extension().string())).string();
		writer.write_mesh(
			contact_path,
			collision_mesh.rest_positions(),
			collision_mesh.dim() == 3 ? collision_mesh.faces() : collision_mesh.edges(),
			collision_mesh.dim() == 3 ? CellType::Triangle : CellType::Line);
	}

	void OutGeometryData::save_wire(
		const std::string &name,
		const OutputSpace &space,
		const OutputFieldFunction &output_fields,
		const double t,
		const ExportOptions &opts) const
	{
		if (!space.mesh || !space.geometry_bases)
			return;

		static const std::map<int, Eigen::MatrixXd> empty_polys;
		static const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> empty_polys_3d;

		const std::vector<basis::ElementBases> &gbases = *space.geometry_bases;
		const mesh::Mesh &mesh = *space.mesh;
		const std::map<int, Eigen::MatrixXd> &polys = space.polys ? *space.polys : empty_polys;
		const std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> &polys_3d = space.polys_3d ? *space.polys_3d : empty_polys_3d;
		const Eigen::VectorXi output_orders =
			space.output_orders.size() == mesh.n_elements()
				? space.output_orders
				: Eigen::VectorXi::Ones(mesh.n_elements());

		Eigen::MatrixXd points, discr, local_points;
		Eigen::MatrixXi cells, element_ids, edges;
		build_vis_mesh(mesh, output_orders, gbases, polys, polys_3d, /*boundary_only=*/false, points, cells, element_ids, discr, local_points);
		if (cells.size() > 0)
			igl::edges(cells, edges);
		else
			edges.resize(0, 2);

		std::shared_ptr<paraviewo::ParaviewWriter> tmpw;
		if (opts.use_hdf5)
			tmpw = std::make_shared<paraviewo::HDF5VTUWriter>();
		else
			tmpw = std::make_shared<paraviewo::VTUWriter>();
		paraviewo::ParaviewWriter &writer = *tmpw;

		// Write the solution alias last so it is the default for warp-by-vector.
		OutputSample sample;
		sample.points = points;
		sample.local_points = local_points;
		if (element_ids.cols() > 0)
			sample.element_ids = element_ids.col(0);
		sample.domain = OutputSample::Domain::Wire;
		sample.cell_count = edges.rows();
		sample.time = t;
		add_output_fields(writer, sample, output_fields);

		writer.write_mesh(name, points, edges, CellType::Line);
	}

	void OutGeometryData::save_points(
		const std::string &path,
		const OutputSpace &space,
		const OutputFieldFunction &output_fields,
		const ExportOptions &opts) const
	{
		if (!space.mesh || !space.dirichlet_nodes || !space.dirichlet_nodes_position)
			return;

		const auto &dirichlet_nodes = *space.dirichlet_nodes;
		const auto &dirichlet_nodes_position = *space.dirichlet_nodes_position;
		const mesh::Mesh &mesh = *space.mesh;

		Eigen::MatrixXd b_sidesets(dirichlet_nodes_position.size(), 1);
		b_sidesets.setZero();
		Eigen::MatrixXd points(dirichlet_nodes_position.size(), mesh.dimension());
		std::vector<CellElement> cells(dirichlet_nodes_position.size());

		for (int i = 0; i < dirichlet_nodes_position.size(); ++i)
		{
			const int n_id = dirichlet_nodes[i];
			const auto s_id = mesh.get_node_id(n_id);
			if (s_id > 0)
			{
				b_sidesets(i) = s_id;
			}

			points.row(i) = dirichlet_nodes_position[i];
			cells[i].vertices.push_back(i);
			cells[i].ctype = CellType::Vertex;
		}

		std::shared_ptr<paraviewo::ParaviewWriter> tmpw;
		if (opts.use_hdf5)
			tmpw = std::make_shared<paraviewo::HDF5VTUWriter>();
		else
			tmpw = std::make_shared<paraviewo::VTUWriter>();
		paraviewo::ParaviewWriter &writer = *tmpw;

		if (opts.export_field("sidesets"))
			writer.add_field("sidesets", b_sidesets);

		// Write the solution alias last so it is the default for warp-by-vector.
		OutputSample sample;
		sample.points = points;
		sample.node_ids.resize(dirichlet_nodes.size());
		for (int i = 0; i < dirichlet_nodes.size(); ++i)
			sample.node_ids(i) = dirichlet_nodes[i];
		sample.domain = OutputSample::Domain::Points;
		sample.cell_count = static_cast<int>(cells.size());
		add_output_fields(writer, sample, output_fields);
		writer.write_mesh(path, points, cells);
	}

	void OutGeometryData::save_pvd(
		const std::string &name,
		const std::function<std::string(int)> &vtu_names,
		int time_steps, double t0, double dt, int skip_frame) const
	{
		paraviewo::PVDWriter::save_pvd(name, vtu_names, time_steps, t0, dt, skip_frame);
	}

	void OutGeometryData::init_sampler(const polyfem::mesh::Mesh &mesh, const double vismesh_rel_area)
	{
		ref_element_sampler.init(mesh.is_volume(), mesh.n_elements(), vismesh_rel_area);
	}

	void OutGeometryData::build_grid(const polyfem::mesh::Mesh &mesh, const double spacing)
	{
		if (spacing <= 0)
			return;

		RowVectorNd min, max;
		mesh.bounding_box(min, max);
		const RowVectorNd delta = max - min;
		const int nx = delta[0] / spacing + 1;
		const int ny = delta[1] / spacing + 1;
		const int nz = delta.cols() >= 3 ? (delta[2] / spacing + 1) : 1;
		const int n = nx * ny * nz;

		grid_points.resize(n, delta.cols());
		int index = 0;
		for (int i = 0; i < nx; ++i)
		{
			const double x = (delta[0] / (nx - 1)) * i + min[0];

			for (int j = 0; j < ny; ++j)
			{
				const double y = (delta[1] / (ny - 1)) * j + min[1];

				if (delta.cols() <= 2)
				{
					grid_points.row(index++) << x, y;
				}
				else
				{
					for (int k = 0; k < nz; ++k)
					{
						const double z = (delta[2] / (nz - 1)) * k + min[2];
						grid_points.row(index++) << x, y, z;
					}
				}
			}
		}

		assert(index == n);

		std::vector<std::array<Eigen::Vector3d, 2>> boxes;
		mesh.elements_boxes(boxes);

		SimpleBVH::BVH bvh;
		bvh.init(boxes);

		const double eps = 1e-6;

		grid_points_to_elements.resize(grid_points.rows(), 1);
		grid_points_to_elements.setConstant(-1);

		grid_points_bc.resize(grid_points.rows(), mesh.is_volume() ? 4 : 3);

		for (int i = 0; i < grid_points.rows(); ++i)
		{
			const Eigen::Vector3d min(
				grid_points(i, 0) - eps,
				grid_points(i, 1) - eps,
				(mesh.is_volume() ? grid_points(i, 2) : 0) - eps);

			const Eigen::Vector3d max(
				grid_points(i, 0) + eps,
				grid_points(i, 1) + eps,
				(mesh.is_volume() ? grid_points(i, 2) : 0) + eps);

			std::vector<unsigned int> candidates;

			bvh.intersect_box(min, max, candidates);

			for (const auto cand : candidates)
			{
				if (!mesh.is_simplex(cand))
				{
					logger().warn("Element {} is not simplex, skipping", cand);
					continue;
				}

				Eigen::MatrixXd coords;
				mesh.barycentric_coords(grid_points.row(i), cand, coords);

				for (int d = 0; d < coords.size(); ++d)
				{
					if (fabs(coords(d)) < 1e-8)
						coords(d) = 0;
					else if (fabs(coords(d) - 1) < 1e-8)
						coords(d) = 1;
				}

				if (coords.array().minCoeff() >= 0 && coords.array().maxCoeff() <= 1)
				{
					grid_points_to_elements(i) = cand;
					grid_points_bc.row(i) = coords;
					break;
				}
			}
		}
	}

	void OutStatsData::compute_mesh_size(const polyfem::mesh::Mesh &mesh_in, const std::vector<polyfem::basis::ElementBases> &bases_in, const int n_samples, const bool use_curved_mesh_size)
	{
		Eigen::MatrixXd samples_simplex, samples_cube, mapped, p0, p1, p;

		mesh_size = 0;
		average_edge_length = 0;
		min_edge_length = std::numeric_limits<double>::max();

		if (!use_curved_mesh_size)
		{
			mesh_in.get_edges(p0, p1);
			p = p0 - p1;
			min_edge_length = p.rowwise().norm().minCoeff();
			average_edge_length = p.rowwise().norm().mean();
			mesh_size = p.rowwise().norm().maxCoeff();

			logger().info("hmin: {}", min_edge_length);
			logger().info("hmax: {}", mesh_size);
			logger().info("havg: {}", average_edge_length);

			return;
		}

		if (mesh_in.is_volume())
		{
			utils::EdgeSampler::sample_3d_simplex(n_samples, samples_simplex);
			utils::EdgeSampler::sample_3d_cube(n_samples, samples_cube);
		}
		else
		{
			utils::EdgeSampler::sample_2d_simplex(n_samples, samples_simplex);
			utils::EdgeSampler::sample_2d_cube(n_samples, samples_cube);
		}

		int n = 0;
		for (size_t i = 0; i < bases_in.size(); ++i)
		{
			if (mesh_in.is_polytope(i))
				continue;
			int n_edges;

			if (mesh_in.is_simplex(i))
			{
				n_edges = mesh_in.is_volume() ? 6 : 3;
				bases_in[i].eval_geom_mapping(samples_simplex, mapped);
			}
			else
			{
				n_edges = mesh_in.is_volume() ? 12 : 4;
				bases_in[i].eval_geom_mapping(samples_cube, mapped);
			}

			for (int j = 0; j < n_edges; ++j)
			{
				double current_edge = 0;
				for (int k = 0; k < n_samples - 1; ++k)
				{
					p0 = mapped.row(j * n_samples + k);
					p1 = mapped.row(j * n_samples + k + 1);
					p = p0 - p1;

					current_edge += p.norm();
				}

				mesh_size = std::max(current_edge, mesh_size);
				min_edge_length = std::min(current_edge, min_edge_length);
				average_edge_length += current_edge;
				++n;
			}
		}

		average_edge_length /= n;

		logger().info("hmin: {}", min_edge_length);
		logger().info("hmax: {}", mesh_size);
		logger().info("havg: {}", average_edge_length);
	}

	void OutStatsData::reset()
	{
		*this = OutStatsData();
	}

	void OutStatsData::count_flipped_elements(const polyfem::mesh::Mesh &mesh, const std::vector<polyfem::basis::ElementBases> &gbases)
	{
		using namespace mesh;

		logger().info("Counting flipped elements...");
		const auto &els_tag = mesh.elements_tag();

		// flipped_elements.clear();
		for (size_t i = 0; i < gbases.size(); ++i)
		{
			if (mesh.is_polytope(i))
				continue;

			polyfem::assembler::ElementAssemblyValues vals;
			if (!vals.is_geom_mapping_positive(mesh.is_volume(), gbases[i]))
			{
				++n_flipped;

				static const std::vector<std::string> element_type_names{{
					"Simplex",
					"RegularInteriorCube",
					"RegularBoundaryCube",
					"SimpleSingularInteriorCube",
					"MultiSingularInteriorCube",
					"SimpleSingularBoundaryCube",
					"InterfaceCube",
					"MultiSingularBoundaryCube",
					"BoundaryPolytope",
					"InteriorPolytope",
					"Undefined",
				}};

				log_and_throw_error("element {} is flipped, type {}", i, element_type_names[static_cast<int>(els_tag[i])]);
			}
		}

		logger().info(" done");

		// dynamic_cast<Mesh3D *>(mesh.get())->save({56}, 1, "mesh.HYBRID");

		// std::sort(flipped_elements.begin(), flipped_elements.end());
		// auto it = std::unique(flipped_elements.begin(), flipped_elements.end());
		// flipped_elements.resize(std::distance(flipped_elements.begin(), it));
	}

	void OutStatsData::compute_errors(
		const int n_bases,
		const std::vector<polyfem::basis::ElementBases> &bases,
		const std::vector<polyfem::basis::ElementBases> &gbases,
		const polyfem::mesh::Mesh &mesh,
		const assembler::Problem &problem,
		const double tend,
		const Eigen::MatrixXd &sol)
	{
		if (n_bases <= 0)
		{
			logger().error("Build the bases first!");
			return;
		}
		if (sol.size() <= 0)
		{
			logger().error("Solve the problem first!");
			return;
		}

		int actual_dim = 1;
		if (!problem.is_scalar())
			actual_dim = mesh.dimension();

		igl::Timer timer;
		timer.start();
		logger().info("Computing errors...");
		using std::max;

		const int n_el = int(bases.size());

		Eigen::MatrixXd v_exact, v_approx;
		Eigen::MatrixXd v_exact_grad(0, 0), v_approx_grad;

		l2_err = 0;
		h1_err = 0;
		grad_max_err = 0;
		h1_semi_err = 0;
		linf_err = 0;
		lp_err = 0;
		// double pred_norm = 0;

		static const int p = 8;

		// Eigen::MatrixXd err_per_el(n_el, 5);
		polyfem::assembler::ElementAssemblyValues vals;

		for (int e = 0; e < n_el; ++e)
		{
			vals.compute(e, mesh.is_volume(), bases[e], gbases[e]);

			if (problem.has_exact_sol())
			{
				problem.exact(vals.val, tend, v_exact);
				problem.exact_grad(vals.val, tend, v_exact_grad);
			}

			v_approx.resize(vals.val.rows(), actual_dim);
			v_approx.setZero();

			v_approx_grad.resize(vals.val.rows(), mesh.dimension() * actual_dim);
			v_approx_grad.setZero();

			const int n_loc_bases = int(vals.basis_values.size());

			for (int i = 0; i < n_loc_bases; ++i)
			{
				const auto &val = vals.basis_values[i];

				for (size_t ii = 0; ii < val.global.size(); ++ii)
				{
					for (int d = 0; d < actual_dim; ++d)
					{
						v_approx.col(d) += val.global[ii].val * sol(val.global[ii].index * actual_dim + d) * val.val;
						v_approx_grad.block(0, d * val.grad_t_m.cols(), v_approx_grad.rows(), val.grad_t_m.cols()) += val.global[ii].val * sol(val.global[ii].index * actual_dim + d) * val.grad_t_m;
					}
				}
			}

			const auto err = problem.has_exact_sol() ? (v_exact - v_approx).eval().rowwise().norm().eval() : (v_approx).eval().rowwise().norm().eval();
			const auto err_grad = problem.has_exact_sol() ? (v_exact_grad - v_approx_grad).eval().rowwise().norm().eval() : (v_approx_grad).eval().rowwise().norm().eval();

			// for(long i = 0; i < err.size(); ++i)
			// errors.push_back(err(i));

			linf_err = std::max(linf_err, err.maxCoeff());
			grad_max_err = std::max(linf_err, err_grad.maxCoeff());

			// {
			// 	const auto &mesh3d = *dynamic_cast<Mesh3D *>(mesh.get());
			// 	const auto v0 = mesh3d.point(mesh3d.cell_vertex(e, 0));
			// 	const auto v1 = mesh3d.point(mesh3d.cell_vertex(e, 1));
			// 	const auto v2 = mesh3d.point(mesh3d.cell_vertex(e, 2));
			// 	const auto v3 = mesh3d.point(mesh3d.cell_vertex(e, 3));

			// 	Eigen::Matrix<double, 6, 3> ee;
			// 	ee.row(0) = v0 - v1;
			// 	ee.row(1) = v1 - v2;
			// 	ee.row(2) = v2 - v0;

			// 	ee.row(3) = v0 - v3;
			// 	ee.row(4) = v1 - v3;
			// 	ee.row(5) = v2 - v3;

			// 	Eigen::Matrix<double, 6, 1> en = ee.rowwise().norm();

			// 	// Eigen::Matrix<double, 3*4, 1> alpha;
			// 	// alpha(0) = angle3(e.row(0), -e.row(1));	 	alpha(1) = angle3(e.row(1), -e.row(2));	 	alpha(2) = angle3(e.row(2), -e.row(0));
			// 	// alpha(3) = angle3(e.row(0), -e.row(4));	 	alpha(4) = angle3(e.row(4), e.row(3));	 	alpha(5) = angle3(-e.row(3), -e.row(0));
			// 	// alpha(6) = angle3(-e.row(4), -e.row(1));	alpha(7) = angle3(e.row(1), -e.row(5));	 	alpha(8) = angle3(e.row(5), e.row(4));
			// 	// alpha(9) = angle3(-e.row(2), -e.row(5));	alpha(10) = angle3(e.row(5), e.row(3));		alpha(11) = angle3(-e.row(3), e.row(2));

			// 	const double S = (ee.row(0).cross(ee.row(1)).norm() + ee.row(0).cross(ee.row(4)).norm() + ee.row(4).cross(ee.row(1)).norm() + ee.row(2).cross(ee.row(5)).norm()) / 2;
			// 	const double V = std::abs(ee.row(3).dot(ee.row(2).cross(-ee.row(0))))/6;
			// 	const double rho = 3 * V / S;
			// 	const double hp = en.maxCoeff();
			// 	const int pp = disc_orders(e);
			// 	const int p_ref = args["space"]["discr_order"];

			// 	err_per_el(e, 0) = err.mean();
			// 	err_per_el(e, 1) = err.maxCoeff();
			// 	err_per_el(e, 2) = std::pow(hp, pp+1)/(rho/hp); // /std::pow(average_edge_length, p_ref+1) * (sqrt(6)/12);
			// 	err_per_el(e, 3) = rho/hp;
			// 	err_per_el(e, 4) = (vals.det.array() * vals.quadrature.weights.array()).sum();

			// 	// pred_norm += (pow(std::pow(hp, pp+1)/(rho/hp),p) * vals.det.array() * vals.quadrature.weights.array()).sum();
			// }

			l2_err += (err.array() * err.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			h1_err += (err_grad.array() * err_grad.array() * vals.det.array() * vals.quadrature.weights.array()).sum();
			lp_err += (err.array().pow(p) * vals.det.array() * vals.quadrature.weights.array()).sum();
		}

		h1_semi_err = sqrt(fabs(h1_err));
		h1_err = sqrt(fabs(l2_err) + fabs(h1_err));
		l2_err = sqrt(fabs(l2_err));

		lp_err = pow(fabs(lp_err), 1. / p);

		// pred_norm = pow(fabs(pred_norm), 1./p);

		timer.stop();
		const double computing_errors_time = timer.getElapsedTime();
		logger().info(" took {}s", computing_errors_time);

		logger().info("-- L2 error: {}", l2_err);
		logger().info("-- Lp error: {}", lp_err);
		logger().info("-- H1 error: {}", h1_err);
		logger().info("-- H1 semi error: {}", h1_semi_err);
		// logger().info("-- Perd norm: {}", pred_norm);

		logger().info("-- Linf error: {}", linf_err);
		logger().info("-- grad max error: {}", grad_max_err);

		// {
		// 	std::ofstream out("errs.txt");
		// 	out<<err_per_el;
		// 	out.close();
		// }
	}

	void OutStatsData::compute_mesh_stats(const polyfem::mesh::Mesh &mesh)
	{
		using namespace polyfem::mesh;

		simplex_count = 0;
		prism_count = 0;
		pyramid_count = 0;
		regular_count = 0;
		regular_boundary_count = 0;
		simple_singular_count = 0;
		multi_singular_count = 0;
		boundary_count = 0;
		non_regular_boundary_count = 0;
		non_regular_count = 0;
		undefined_count = 0;
		multi_singular_boundary_count = 0;

		const auto &els_tag = mesh.elements_tag();

		for (size_t i = 0; i < els_tag.size(); ++i)
		{
			const ElementType type = els_tag[i];

			switch (type)
			{
			case ElementType::SIMPLEX:
				simplex_count++;
				break;
			case ElementType::PRISM:
				prism_count++;
				break;
			case ElementType::PYRAMID:
				pyramid_count++;
				break;
			case ElementType::REGULAR_INTERIOR_CUBE:
				regular_count++;
				break;
			case ElementType::REGULAR_BOUNDARY_CUBE:
				regular_boundary_count++;
				break;
			case ElementType::SIMPLE_SINGULAR_INTERIOR_CUBE:
				simple_singular_count++;
				break;
			case ElementType::MULTI_SINGULAR_INTERIOR_CUBE:
				multi_singular_count++;
				break;
			case ElementType::SIMPLE_SINGULAR_BOUNDARY_CUBE:
				boundary_count++;
				break;
			case ElementType::INTERFACE_CUBE:
			case ElementType::MULTI_SINGULAR_BOUNDARY_CUBE:
				multi_singular_boundary_count++;
				break;
			case ElementType::BOUNDARY_POLYTOPE:
				non_regular_boundary_count++;
				break;
			case ElementType::INTERIOR_POLYTOPE:
				non_regular_count++;
				break;
			case ElementType::UNDEFINED:
				undefined_count++;
				break;
			default:
				throw std::runtime_error("Unknown element type");
			}
		}

		logger().info("simplex_count: \t{}", simplex_count);
		logger().info("prism_count: \t{}", prism_count);
		logger().info("pyramid_count: \t{}", pyramid_count);
		logger().info("regular_count: \t{}", regular_count);
		logger().info("regular_boundary_count: \t{}", regular_boundary_count);
		logger().info("simple_singular_count: \t{}", simple_singular_count);
		logger().info("multi_singular_count: \t{}", multi_singular_count);
		logger().info("boundary_count: \t{}", boundary_count);
		logger().info("multi_singular_boundary_count: \t{}", multi_singular_boundary_count);
		logger().info("non_regular_count: \t{}", non_regular_count);
		logger().info("non_regular_boundary_count: \t{}", non_regular_boundary_count);
		logger().info("undefined_count: \t{}", undefined_count);
		logger().info("total count:\t {}", mesh.n_elements());
	}

	void OutStatsData::save_json(
		const nlohmann::json &args,
		const int n_bases, const int n_pressure_bases,
		const Eigen::MatrixXd &sol,
		const mesh::Mesh &mesh,
		const Eigen::VectorXi &disc_orders,
		const Eigen::VectorXi &disc_ordersq,
		const assembler::Problem &problem,
		const OutRuntimeData &runtime,
		const std::string &formulation,
		const bool isoparametric,
		const int sol_at_node_id,
		nlohmann::json &j) const
	{

		j["args"] = args;

		j["geom_order"] = mesh.orders().size() > 0 ? mesh.orders().maxCoeff() : 1;
		j["geom_order_min"] = mesh.orders().size() > 0 ? mesh.orders().minCoeff() : 1;
		j["discr_order_min"] = disc_orders.minCoeff();
		j["discr_order_max"] = disc_orders.maxCoeff();
		j["discr_orderq_min"] = disc_ordersq.minCoeff();
		j["discr_orderq_max"] = disc_ordersq.maxCoeff();
		j["iso_parametric"] = isoparametric;
		j["problem"] = problem.name();
		j["mat_size"] = mat_size;
		j["num_bases"] = n_bases;
		j["num_pressure_bases"] = n_pressure_bases;
		j["num_non_zero"] = nn_zero;
		j["num_flipped"] = n_flipped;
		j["num_dofs"] = num_dofs;
		j["num_vertices"] = mesh.n_vertices();
		j["num_elements"] = mesh.n_elements();

		j["num_p1"] = (disc_orders.array() == 1).count();
		j["num_p2"] = (disc_orders.array() == 2).count();
		j["num_p3"] = (disc_orders.array() == 3).count();
		j["num_p4"] = (disc_orders.array() == 4).count();
		j["num_p5"] = (disc_orders.array() == 5).count();

		j["mesh_size"] = mesh_size;
		j["max_angle"] = max_angle;

		j["sigma_max"] = sigma_max;
		j["sigma_min"] = sigma_min;
		j["sigma_avg"] = sigma_avg;

		j["min_edge_length"] = min_edge_length;
		j["average_edge_length"] = average_edge_length;

		j["err_l2"] = l2_err;
		j["err_h1"] = h1_err;
		j["err_h1_semi"] = h1_semi_err;
		j["err_linf"] = linf_err;
		j["err_linf_grad"] = grad_max_err;
		j["err_lp"] = lp_err;

		j["spectrum"] = {spectrum(0), spectrum(1), spectrum(2), spectrum(3)};
		j["spectrum_condest"] = std::abs(spectrum(3)) / std::abs(spectrum(0));

		// j["errors"] = errors;

		j["time_building_basis"] = runtime.building_basis_time;
		j["time_loading_mesh"] = runtime.loading_mesh_time;
		j["time_computing_poly_basis"] = runtime.computing_poly_basis_time;
		j["time_assembling_stiffness_mat"] = runtime.assembling_stiffness_mat_time;
		j["time_assembling_mass_mat"] = runtime.assembling_mass_mat_time;
		j["time_assigning_rhs"] = runtime.assigning_rhs_time;
		j["time_solving"] = runtime.solving_time;
		// j["time_computing_errors"] = runtime.computing_errors_time;

		j["solver_info"] = solver_info;

		j["count_simplex"] = simplex_count;
		j["count_prism"] = prism_count;
		j["count_pyramid"] = pyramid_count;
		j["count_regular"] = regular_count;
		j["count_regular_boundary"] = regular_boundary_count;
		j["count_simple_singular"] = simple_singular_count;
		j["count_multi_singular"] = multi_singular_count;
		j["count_boundary"] = boundary_count;
		j["count_non_regular_boundary"] = non_regular_boundary_count;
		j["count_non_regular"] = non_regular_count;
		j["count_undefined"] = undefined_count;
		j["count_multi_singular_boundary"] = multi_singular_boundary_count;

		j["is_simplicial"] = mesh.n_elements() == simplex_count;

		j["peak_memory"] = getPeakRSS() / (1024 * 1024);

		const int actual_dim = problem.is_scalar() ? 1 : mesh.dimension();

		std::vector<double> mmin(actual_dim);
		std::vector<double> mmax(actual_dim);

		for (int d = 0; d < actual_dim; ++d)
		{
			mmin[d] = std::numeric_limits<double>::max();
			mmax[d] = -std::numeric_limits<double>::max();
		}

		for (int i = 0; i < sol.size(); i += actual_dim)
		{
			for (int d = 0; d < actual_dim; ++d)
			{
				mmin[d] = std::min(mmin[d], sol(i + d));
				mmax[d] = std::max(mmax[d], sol(i + d));
			}
		}

		std::vector<double> sol_at_node(actual_dim);

		if (sol_at_node_id >= 0)
		{
			const int node_id = sol_at_node_id;

			for (int d = 0; d < actual_dim; ++d)
			{
				sol_at_node[d] = sol(node_id * actual_dim + d);
			}
		}

		j["sol_at_node"] = sol_at_node;
		j["sol_min"] = mmin;
		j["sol_max"] = mmax;

#if defined(POLYFEM_WITH_CPP_THREADS)
		j["num_threads"] = utils::get_n_threads();
#elif defined(POLYFEM_WITH_TBB)
		j["num_threads"] = utils::get_n_threads();
#else
		j["num_threads"] = 1;
#endif

		j["formulation"] = formulation;

		logger().info("done");
	}

} // namespace polyfem::io

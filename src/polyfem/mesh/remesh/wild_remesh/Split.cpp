#include <polyfem/mesh/remesh/WildRemesh2D.hpp>

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/GeometryUtils.hpp>

#include <wmtk/utils/ExecutorUtils.hpp>

namespace polyfem::mesh
{
	bool WildRemeshing2D::split_edge_before(const Tuple &t)
	{
		if (!super::split_edge_before(t))
			return false;

		const VertexAttributes &e0 = vertex_attrs[t.vid(*this)];
		const VertexAttributes &e1 = vertex_attrs[t.switch_vertex(*this).vid(*this)];

		// Dont split if the edge is too small
		if ((e1.position - e0.position).norm() < 1e-6)
			return false;

		// if (e0.fixed && e1.fixed)
		// 	return false;

		edge_cache = EdgeCache(*this, t);

		energy_before = compute_global_energy();
		if (!std::isfinite(energy_before))
			return false;
		// energy_before = compute_global_wicke_measure();

		return true;
	}

	bool WildRemeshing2D::split_edge_after(const Tuple &t)
	{
		const auto &[v0, v1, old_edges, old_faces] = edge_cache;

		const size_t new_vid = t.vid(*this);
		// const std::vector<Tuple> new_faces = get_one_ring_tris_for_vertex(t);
		// const std::vector<Tuple> new_edges = new_edges_after(new_faces);

		vertex_attrs[new_vid] = {
			(v0.rest_position + v1.rest_position) / 2.0,
			(v0.position + v1.position) / 2.0,
			(v0.projection_quantities + v1.projection_quantities) / 2.0,
			v0.fixed && v1.fixed,
			v0.partition_id,
		};

		// Assign edge attributes to the new edges
		Tuple nav = t.switch_face(*this)->switch_edge(*this);
		edge_attrs[nav.eid(*this)] = old_edges[0];

		nav = nav.switch_vertex(*this).switch_edge(*this);
		edge_attrs[nav.eid(*this)] = old_edges[1];

		nav = nav.switch_vertex(*this).switch_edge(*this);
		edge_attrs[nav.eid(*this)].boundary_id = -1; // interior edge

		nav = nav.switch_face(*this)->switch_edge(*this);
		edge_attrs[nav.eid(*this)] = old_edges[2];
		nav = nav.switch_vertex(*this).switch_edge(*this);
		edge_attrs[nav.eid(*this)] = old_edges[0];

		if (nav.switch_face(*this))
		{
			nav = nav.switch_face(*this)->switch_edge(*this);
			edge_attrs[nav.eid(*this)] = old_edges[3];

			nav = nav.switch_vertex(*this).switch_edge(*this);
			edge_attrs[nav.eid(*this)].boundary_id = -1; // interior edge

			nav = nav.switch_face(*this)->switch_edge(*this);
			edge_attrs[nav.eid(*this)] = old_edges[4];
#ifndef NDEBUG
			nav = nav.switch_vertex(*this).switch_edge(*this);
			assert(edge_attrs[nav.eid(*this)].boundary_id == old_edges[0].boundary_id);
#endif
		}

		// Assign face attributes to the new faces
		nav = t.switch_face(*this).value();
		face_attrs[nav.fid(*this)] = old_faces[0];
		nav = nav.switch_face(*this).value();
		face_attrs[nav.fid(*this)] = old_faces[0];
		nav = nav.switch_edge(*this);
		if (nav.switch_face(*this))
		{
			nav = nav.switch_face(*this).value();
			face_attrs[nav.fid(*this)] = old_faces[1];
			nav = nav.switch_edge(*this).switch_face(*this).value();
			face_attrs[nav.fid(*this)] = old_faces[1];

#ifndef NDEBUG
			nav = nav.switch_edge(*this).switch_face(*this).value();
			assert(face_attrs[nav.fid(*this)].body_id == old_faces[0].body_id);
#endif
		}

		// double energy_after = compute_global_energy();
		// double energy_after = compute_global_wicke_measure();

		// logger().critical("energy_before={} energy_after={} accept={}", energy_before, energy_after, energy_after < energy_before);
		// return energy_after < energy_before;
		// return energy_after < energy_before - 1e-14;
		const double rel_energy = local_relaxation(t, 3);
		logger().critical("rel_energy={}", rel_energy);
		return rel_energy <= -1e-4; // accept if energy decreased by at least 1%
	}

	void WildRemeshing2D::split_all_edges()
	{
		using Operations = std::vector<std::pair<std::string, Tuple>>;

		write_rest_obj("rest_mesh_before.obj");
		write_deformed_obj("deformed_mesh_before.obj");

		Operations collect_all_ops;
		for (const Tuple &e : get_edges())
		{
			collect_all_ops.emplace_back("edge_split", e);
		}

		wmtk::ExecutePass<WildRemeshing2D, EXECUTION_POLICY> executor;
		// if (NUM_THREADS > 0)
		// {
		// 	executor.lock_vertices = [&](WildRemeshing2D &m, const Tuple &e, int task_id) -> bool {
		// 		return m.try_set_edge_mutex_two_ring(e, task_id);
		// 	};
		// 	executor.num_threads = NUM_THREADS;
		// }

		using namespace assembler;
		NeoHookeanElasticity neo_hookean;
		neo_hookean.set_size(DIM);
		// TODO: set the material parameters
		neo_hookean.add_multimaterial(0, R"({
			"id": 1,
			"E": 3500,
			"nu": 0.4,
			"rho": 1000,
			"type": "NeoHookean"
		})"_json);

		executor.priority = [&neo_hookean](const WildRemeshing2D &m, std::string op, const Tuple &t) -> double {
			// NOTE: this code compute the edge length
			// return (m.vertex_attrs[t.vid(m)].position
			// 		- m.vertex_attrs[t.switch_vertex(m).vid(m)].position)
			// 	.squaredNorm();

			Eigen::MatrixXd V, U;
			Eigen::MatrixXi F;
			std::unordered_map<size_t, size_t> vi_map;
			std::vector<int> body_ids;
			{
				std::vector<Tuple> tris{{t}};
				if (t.switch_face(m))
					tris.push_back(t.switch_face(m).value());
				m.build_local_matricies(tris, V, U, F, vi_map, body_ids);
			}

			std::vector<polyfem::basis::ElementBases> bases;
			Eigen::VectorXi vertex_to_basis;
			WildRemeshing2D::build_bases(
				V, F, m.state.formulation(), bases, vertex_to_basis);

			Eigen::VectorXd displacements = utils::flatten(utils::reorder_matrix(U, vertex_to_basis));

			AssemblyValsCache cache;
			// TODO: set the material parameters
			const double energy = m.state.assembler.assemble_energy(
				m.state.formulation(),
				/*is_volume=*/DIM == 3,
				bases,
				/*gbases=*/bases,
				cache,
				/*dt=*/-1,
				displacements,
				/*displacement_prev=*/Eigen::MatrixXd());
			assert(std::isfinite(energy));
			return energy / F.rows(); // average energy per face

			// double max_stress = -std::numeric_limits<double>::infinity();
			// for (int el_id = 0; el_id < F.rows(); el_id++)
			// {
			// 	Eigen::MatrixXd local_pts(1, DIM);
			// 	local_pts << 1 / 3.0, 1 / 3.0;

			// 	Eigen::MatrixXd stress(1, 1);
			// 	m.state.assembler.compute_scalar_value(
			// 		m.assembler_formulation,
			// 		el_id,
			// 		bases[el_id],
			// 		/*gbases=*/bases[el_id],
			// 		local_pts,
			// 		displacements,
			// 		stress);
			// 	stress *= utils::triangle_area_2D(
			// 		V.row(F(el_id, 0)), V.row(F(el_id, 1)), V.row(F(el_id, 2)));

			// 	max_stress = std::max(max_stress, stress(0));
			// }
			// assert(std::isfinite(max_stress));
			// return max_stress;
		};

		executor.renew_neighbor_tuples = [](const WildRemeshing2D &m, std::string op, const std::vector<Tuple> &tris) -> Operations {
			auto edges = m.new_edges_after(tris);
			Operations new_ops;
			for (auto &e : edges)
				new_ops.emplace_back("edge_split", e);
			return new_ops;
		};

		// Split 25% of edges
		int num_splits = 0;
		const int max_splits = std::round(0.2 * collect_all_ops.size());
		executor.stopping_criterion = [&](const WildRemeshing2D &m) -> bool {
			return (++num_splits) > max_splits;
		};
		executor.stopping_criterion_checking_frequency = 1;

		executor(*this, collect_all_ops);

		write_rest_obj("rest_mesh_after.obj");
		write_deformed_obj("deformed_mesh_after.obj");
	}

} // namespace polyfem::mesh
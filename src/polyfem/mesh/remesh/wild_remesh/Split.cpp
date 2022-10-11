#include <polyfem/mesh/remesh/WildRemesh2D.hpp>

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/assembler/NeoHookeanElasticity.hpp>

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
		// if ((e1.position - e0.position).norm() < 5e-3)
		// 	return false;

		// if (e0.frozen && e1.frozen)
		// 	return false;

		edge_cache = EdgeCache(*this, t);

		energy_before = compute_global_energy();
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
			.rest_position = (v0.rest_position + v1.rest_position) / 2.0,
			.position = (v0.position + v1.position) / 2.0,
			.velocity = (v0.velocity + v1.velocity) / 2.0,
			.acceleration = (v0.acceleration + v1.acceleration) / 2.0,
			.partition_id = v0.partition_id,
			.frozen = v0.frozen && v1.frozen,
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

		double energy_after = compute_global_energy();
		// double energy_after = compute_global_wicke_measure();

		// logger().critical("energy_before={} energy_after={} accept={}", energy_before, energy_after, energy_after < energy_before);
		// return energy_after < energy_before;
		// return energy_after < energy_before - 1e-14;
		return true;
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

			std::vector<polyfem::basis::ElementBases> bases;
			Eigen::VectorXi vertex_to_basis;
			Eigen::MatrixXi F;
			{
				const std::array<Tuple, 3> vs = m.oriented_tri_vertices(t);

				F.resize(1, 3);
				F << vs[0].vid(m), vs[1].vid(m), vs[2].vid(m);

				Eigen::MatrixXd V(3, DIM);
				V.row(0) = m.vertex_attrs[F(0, 0)].position;
				V.row(1) = m.vertex_attrs[F(0, 1)].position;
				V.row(2) = m.vertex_attrs[F(0, 2)].position;

				WildRemeshing2D::build_bases(V, F, bases, vertex_to_basis);
			}

			ElementAssemblyValues vals;
			vals.compute(
				/*el_index=*/0, /*is_volume=*/DIM == 3, bases[0], /*gbases=*/bases[0]);

			assert(MAX_QUAD_POINTS == -1 || vals.quadrature.weights.size() < MAX_QUAD_POINTS);
			QuadratureVector da = vals.det.array() * vals.quadrature.weights.array();

			Eigen::VectorXd displacements(3 * DIM);
			for (int i = 0; i < 3; i++)
				displacements.segment<DIM>(vertex_to_basis[i] * DIM) =
					m.vertex_attrs[F(0, i)].displacement();

			const double energy = neo_hookean.compute_energy(NonLinearAssemblerData(
				vals, /*dt=*/-1, displacements, /*displacement_prev=*/Eigen::MatrixXd(), da));
			assert(std::isfinite(energy));
			return energy;
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
		const int max_splits = std::round(1.0 * collect_all_ops.size());
		executor.stopping_criterion = [&](const WildRemeshing2D &m) -> bool {
			return (++num_splits) > max_splits;
		};
		executor.stopping_criterion_checking_frequency = 1;

		executor(*this, collect_all_ops);

		write_rest_obj("rest_mesh_after.obj");
		write_deformed_obj("deformed_mesh_after.obj");
	}

} // namespace polyfem::mesh
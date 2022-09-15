#include "WildRemesh2D.hpp"

#include <polyfem/mesh/remesh/wild_remesh/AMIPSForm.hpp>

// #include <polyfem/utils/Logger.hpp>

// #include <wmtk/ExecutionScheduler.hpp>
// #include <wmtk/utils/TriQualityUtils.hpp>
// #include <wmtk/utils/TupleUtils.hpp>
// #include <wmtk/utils/AMIPS2D.h>

#include <igl/boundary_facets.h>
#include <igl/predicates/predicates.h>
#include <igl/writeOBJ.h>

namespace polyfem::mesh
{
	void WildRemeshing2D::create_mesh(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const Eigen::MatrixXd &displacements,
		const Eigen::MatrixXd &velocities,
		const Eigen::MatrixXd &accelerations)
	{
		Eigen::MatrixXi BE;
		igl::boundary_facets(F, BE);

		std::vector<bool> is_boundary_vertex(V.rows(), false);
		for (int i = 0; i < BE.rows(); ++i)
		{
			is_boundary_vertex[BE(i, 0)] = true;
			is_boundary_vertex[BE(i, 1)] = true;
		}

		// Register attributes
		p_vertex_attrs = &vertex_attrs;

		// Convert from eigen to internal representation (TODO: move to utils and remove it from all app)
		std::vector<std::array<size_t, 3>> tri(F.rows());

		for (int i = 0; i < F.rows(); i++)
			for (int j = 0; j < 3; j++)
				tri[i][j] = (size_t)F(i, j);

		// Initialize the trimesh class which handles connectivity
		wmtk::TriMesh::create_mesh(V.rows(), tri);

		// Save the vertex position in the vertex attributes
		for (unsigned i = 0; i < V.rows(); ++i)
		{
			vertex_attrs[i].rest_position = V.row(i).head<2>();
			vertex_attrs[i].displacement = displacements.row(i).head<2>();
			vertex_attrs[i].velocity = velocities.row(i).head<2>();
			vertex_attrs[i].acceleration = accelerations.row(i).head<2>();
			vertex_attrs[i].frozen = is_boundary_vertex[i];
		}
	}

	void WildRemeshing2D::export_mesh(
		Eigen::MatrixXd &V,
		Eigen::MatrixXi &F,
		Eigen::MatrixXd &displacements,
		Eigen::MatrixXd &velocities,
		Eigen::MatrixXd &accelerations)
	{
		V = Eigen::MatrixXd::Zero(vert_capacity(), 2);
		displacements = Eigen::MatrixXd::Zero(vert_capacity(), 2);
		velocities = Eigen::MatrixXd::Zero(vert_capacity(), 2);
		accelerations = Eigen::MatrixXd::Zero(vert_capacity(), 2);
		for (const Tuple &t : get_vertices())
		{
			const size_t i = t.vid(*this);
			V.row(i) = vertex_attrs[i].rest_position;
			displacements.row(i) = vertex_attrs[i].displacement;
			velocities.row(i) = vertex_attrs[i].velocity;
			accelerations.row(i) = vertex_attrs[i].acceleration;
		}

		F = Eigen::MatrixXi::Constant(tri_capacity(), 3, -1);
		for (const Tuple &t : get_faces())
		{
			const size_t i = t.fid(*this);
			const std::array<Tuple, 3> vs = oriented_tri_vertices(t);
			for (int j = 0; j < 3; j++)
			{
				F(i, j) = vs[j].vid(*this);
			}
		}
	}

	void WildRemeshing2D::write_obj(const std::string &path)
	{
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;
		Eigen::MatrixXd displacements;
		Eigen::MatrixXd velocities;
		Eigen::MatrixXd accelerations;

		export_mesh(V, F, displacements, velocities, accelerations);

		Eigen::MatrixXd V3 = Eigen::MatrixXd::Zero(V.rows(), 3);
		V3.leftCols(2) = V;

		igl::writeOBJ(path, V3, F);
	}

	// void WildRemeshing2D::freeze_vertex(TriMesh::Tuple &v)
	// {
	// 	for (auto e : get_one_ring_edges_for_vertex(v))
	// 	{
	// 		if (is_boundary_edge(e))
	// 		{
	// 			vertex_attrs[v.vid(*this)].frozen = true;
	// 			continue;
	// 		}
	// 	}
	// }

	double WildRemeshing2D::get_quality(const Tuple &loc) const
	{
		// Global ids of the vertices of the triangle
		const std::array<size_t, 3> its = super::oriented_tri_vids(loc);

		// Energy evaluation
		const double energy = solver::AMIPSForm::energy(
			vertex_attrs[its[0]].rest_position,
			vertex_attrs[its[1]].rest_position,
			vertex_attrs[its[2]].rest_position,
			vertex_attrs[its[0]].position(),
			vertex_attrs[its[1]].position(),
			vertex_attrs[its[2]].position());

		// Filter for numerical issues
		if (!std::isfinite(energy))
			return MAX_ENERGY;

		return energy;
	}

	Eigen::VectorXd WildRemeshing2D::get_quality_all_triangles()
	{
		// Use a concurrent vector as for_each_face is parallel
		tbb::concurrent_vector<double> quality;
		quality.reserve(vertex_attrs.size());

		// Evaluate quality in parallel
		for_each_face(
			[&](const Tuple &f) {
				quality.push_back(get_quality(f));
			});

		// Copy back in a VectorXd
		Eigen::VectorXd ret(quality.size());
		for (unsigned i = 0; i < quality.size(); ++i)
			ret[i] = quality[i];
		return ret;
	}

	bool WildRemeshing2D::is_inverted(const Tuple &loc) const
	{
		// Get the vertices ids
		const std::array<Tuple, 3> vs = oriented_tri_vertices(loc);

		igl::predicates::exactinit();

		// Use igl for checking orientation
		igl::predicates::Orientation res = igl::predicates::orient2d(
			vertex_attrs[vs[0].vid(*this)].position(),
			vertex_attrs[vs[1].vid(*this)].position(),
			vertex_attrs[vs[2].vid(*this)].position());

		// The element is inverted if it not positive (i.e. it is negative or it is degenerate)
		return (res != igl::predicates::Orientation::POSITIVE);
	}

	// TODO define invariants function

	// bool WildRemeshing2D::collapse_edge_before(const Tuple &t)
	// {
	// 	if (!super::collapse_edge_before(t))
	// 		return false;
	// 	if (is_vertex_frozen(t) && is_vertex_frozen(t.switch_vertex(*this)))
	// 		return false;
	// 	position_cache.local().v1_attr = vertex_attrs[t.vid(*this)];
	// 	position_cache.local().v2_attr = vertex_attrs[t.switch_vertex(*this).vid(*this)];
	// 	return true;
	// }

	// bool WildRemeshing2D::collapse_edge_after(const TriMesh::Tuple &t)
	// {
	// 	const auto &[v1_attr, v2_attr] = position_cache.local();
	// 	VertexAttributes &v_attr = vertex_attrs[t.vid(*this)];
	// 	const int partition_id = v_attr.partition_id;

	// 	if (v1_attr.frozen)
	// 		v_attr = v1_attr;
	// 	else if (v2_attr.frozen)
	// 		v_attr = v2_attr;
	// 	else
	// 	{
	// 		vertex_attrs[t.vid(*this)].rest_position = (v1_attr.rest_position + v2_attr.rest_position) / 2.0;
	// 		vertex_attrs[t.vid(*this)].displacement = (v1_attr.displacement + v2_attr.displacement) / 2.0;
	// 		vertex_attrs[t.vid(*this)].velocity = (v1_attr.velocity + v2_attr.velocity) / 2.0;
	// 		vertex_attrs[t.vid(*this)].acceleration = (v1_attr.acceleration + v2_attr.acceleration) / 2.0;
	// 		vertex_attrs[t.vid(*this)].frozen = false;
	// 	}
	// 	v_attr.partition_id = partition_id;

	// 	return true;
	// }

	// std::vector<wmtk::TriMesh::Tuple> WildRemeshing2D::new_edges_after(
	// 	const std::vector<wmtk::TriMesh::Tuple> &tris) const
	// {
	// 	std::vector<wmtk::TriMesh::Tuple> new_edges;
	// 	std::vector<size_t> one_ring_fid;

	// 	for (auto t : tris)
	// 	{
	// 		for (auto j = 0; j < 3; j++)
	// 		{
	// 			new_edges.push_back(tuple_from_edge(t.fid(*this), j));
	// 		}
	// 	}
	// 	wmtk::unique_edge_tuples(*this, new_edges);
	// 	return new_edges;
	// }

	// bool WildRemeshing2D::collapse_shortest(int target_vert_number)
	// {
	// 	size_t initial_size = get_vertices().size();
	// 	auto collect_all_ops = std::vector<std::pair<std::string, Tuple>>();
	// 	for (auto &loc : get_edges())
	// 		collect_all_ops.emplace_back("edge_collapse", loc);

	// 	auto renew = [](auto &m, auto op, auto &tris) {
	// 		auto edges = m.new_edges_after(tris);
	// 		auto optup = std::vector<std::pair<std::string, Tuple>>();
	// 		for (auto &e : edges)
	// 			optup.emplace_back("edge_collapse", e);
	// 		return optup;
	// 	};
	// 	auto measure_len2 = [](auto &m, auto op, const Tuple &new_e) {
	// 		auto len2 =
	// 			(m.vertex_attrs[new_e.vid(m)].rest_position - m.vertex_attrs[new_e.switch_vertex(m).vid(m)].rest_position)
	// 				.squaredNorm();
	// 		return -len2;
	// 	};
	// 	auto setup_and_execute = [&](auto executor) {
	// 		executor.num_threads = NUM_THREADS;
	// 		executor.renew_neighbor_tuples = renew;
	// 		executor.priority = measure_len2;
	// 		executor.stopping_criterion_checking_frequency =
	// 			target_vert_number > 0 ? (initial_size - target_vert_number - 1)
	// 								   : std::numeric_limits<int>::max();
	// 		executor.stopping_criterion = [](auto &m) { return true; };
	// 		executor(*this, collect_all_ops);
	// 	};

	// 	if (NUM_THREADS > 0)
	// 	{
	// 		auto executor = wmtk::ExecutePass<WildRemeshing2D, wmtk::ExecutionPolicy::kPartition>();
	// 		executor.lock_vertices = [](auto &m, const auto &e, int task_id) {
	// 			return m.try_set_edge_mutex_two_ring(e, task_id);
	// 		};
	// 		setup_and_execute(executor);
	// 	}
	// 	else
	// 	{
	// 		auto executor = wmtk::ExecutePass<WildRemeshing2D, wmtk::ExecutionPolicy::kSeq>();
	// 		setup_and_execute(executor);
	// 	}
	// 	return true;
	// }

} // namespace polyfem::mesh
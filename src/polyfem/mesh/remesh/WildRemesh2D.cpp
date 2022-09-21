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
		const Eigen::MatrixXd &rest_positions,
		const Eigen::MatrixXd &positions,
		const Eigen::MatrixXi &triangles)
	{
		Eigen::MatrixXi boundary_edges;
		igl::boundary_facets(triangles, boundary_edges);

		std::vector<bool> is_boundary_vertex(positions.rows(), false);
		for (int i = 0; i < boundary_edges.rows(); ++i)
		{
			is_boundary_vertex[boundary_edges(i, 0)] = true;
			is_boundary_vertex[boundary_edges(i, 1)] = true;
		}

		// Register attributes
		p_vertex_attrs = &vertex_attrs;

		// Convert from eigen to internal representation (TODO: move to utils and remove it from all app)
		std::vector<std::array<size_t, 3>> tri(triangles.rows());

		for (int i = 0; i < triangles.rows(); i++)
			for (int j = 0; j < 3; j++)
				tri[i][j] = (size_t)triangles(i, j);

		// Initialize the trimesh class which handles connectivity
		wmtk::TriMesh::create_mesh(positions.rows(), tri);

		// Save the vertex position in the vertex attributes
		for (unsigned i = 0; i < positions.rows(); ++i)
		{
			vertex_attrs[i].rest_position = rest_positions.row(i).head<2>();
			vertex_attrs[i].position = positions.row(i).head<2>();
			vertex_attrs[i].frozen = is_boundary_vertex[i];
		}
	}

	void WildRemeshing2D::export_mesh(
		Eigen::MatrixXd &rest_positions,
		Eigen::MatrixXd &positions,
		Eigen::MatrixXi &triangles)
	{
		rest_positions = Eigen::MatrixXd::Zero(vert_capacity(), 2);
		positions = Eigen::MatrixXd::Zero(vert_capacity(), 2);
		for (const Tuple &t : get_vertices())
		{
			const size_t i = t.vid(*this);
			rest_positions.row(i) = vertex_attrs[i].rest_position;
			positions.row(i) = vertex_attrs[i].position;
		}

		triangles = Eigen::MatrixXi::Constant(tri_capacity(), 3, -1);
		for (const Tuple &t : get_faces())
		{
			const size_t i = t.fid(*this);
			const std::array<Tuple, 3> vs = oriented_tri_vertices(t);
			for (int j = 0; j < 3; j++)
			{
				triangles(i, j) = vs[j].vid(*this);
			}
		}
	}

	void WildRemeshing2D::write_rest_obj(const std::string &path)
	{
		Eigen::MatrixXd rest_positions;
		Eigen::MatrixXd _;
		Eigen::MatrixXi triangles;
		export_mesh(rest_positions, _, triangles);

		rest_positions.conservativeResize(rest_positions.rows(), 3);
		rest_positions.col(2).setZero();

		igl::writeOBJ(path, rest_positions, triangles);
	}

	void WildRemeshing2D::write_deformed_obj(const std::string &path)
	{
		Eigen::MatrixXd _;
		Eigen::MatrixXd positions;
		Eigen::MatrixXi triangles;
		export_mesh(_, positions, triangles);

		positions.conservativeResize(positions.rows(), 3);
		positions.col(2).setZero();

		igl::writeOBJ(path, positions, triangles);
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
			vertex_attrs[its[0]].position,
			vertex_attrs[its[1]].position,
			vertex_attrs[its[2]].position);

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
			vertex_attrs[vs[0].vid(*this)].rest_position,
			vertex_attrs[vs[1].vid(*this)].rest_position,
			vertex_attrs[vs[2].vid(*this)].rest_position);

		// The element is inverted if it not positive (i.e. it is negative or it is degenerate)
		return (res != igl::predicates::Orientation::POSITIVE);
	}

	bool WildRemeshing2D::invariants(const std::vector<Tuple> &new_tris)
	{
		for (auto &t : new_tris)
		{
			if (is_inverted(t))
			{
				return false;
			}
		}
		return true;
	}

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
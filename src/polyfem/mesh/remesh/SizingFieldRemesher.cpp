#include "SizingFieldRemesher.hpp"

#include <ipc/collision_mesh.hpp>
#include <ipc/candidates/candidates.hpp>
#include <ipc/broad_phase/hash_grid.hpp>

#include <Eigen/Dense>

namespace polyfem::mesh
{
	// Edge splitting
	template <class WMTKMesh>
	void SizingFieldRemesher<WMTKMesh>::split_edges()
	{
		Operations splits;
		const std::unordered_map<size_t, double> edge_sizings = this->compute_edge_sizings();
		double min = std::numeric_limits<double>::max();
		double max = std::numeric_limits<double>::min();
		for (const Tuple &e : WMTKMesh::get_edges())
		{
			min = std::min(min, edge_sizings.at(e.eid(*this)));
			max = std::max(max, edge_sizings.at(e.eid(*this)));
			if (edge_sizings.at(e.eid(*this)) >= 1)
				splits.emplace_back("edge_split", e);
		}
		logger().debug("min sij: {}, max sij: {}", min, max);

		if (splits.empty())
			return;

		executor.priority = [&](const WildRemesher<WMTKMesh> &m, std::string op, const Tuple &t) -> double {
			return edge_sizings.at(t.eid(m));
		};

		executor(*this, splits);
	}

	// Edge collapse
	template <class WMTKMesh>
	void SizingFieldRemesher<WMTKMesh>::collapse_edges()
	{
		Operations collapses;
		const std::unordered_map<size_t, double> edge_sizings = this->compute_edge_sizings();
		for (const Tuple &e : WMTKMesh::get_edges())
			if (edge_sizings.at(e.eid(*this)) <= 0.8)
				collapses.emplace_back("edge_collapse", e);

		if (collapses.empty())
			return;

		executor.priority = [&](const WildRemesher<WMTKMesh> &m, std::string op, const Tuple &t) -> double {
			// return -edge_sizings.at(t.eid(m));
			return -m.rest_edge_length(t);
		};

		executor(*this, collapses);
	}

	template <class WMTKMesh>
	bool SizingFieldRemesher<WMTKMesh>::split_edge_before(const Tuple &t)
	{
		if (!Super::split_edge_before(t))
			return false;

		// NOTE: this is a hack to avoid splitting edges that are too short
		if (this->rest_edge_length(t) < 0.01)
			return false;

		return true;
	}

	template <class WMTKMesh>
	bool SizingFieldRemesher<WMTKMesh>::collapse_edge_after(const Tuple &t)
	{
		if (!Super::collapse_edge_after(t))
			return false;

		if constexpr (Super::DIM == 2)
		{
			const std::unordered_map<size_t, double> edge_sizings = this->compute_edge_sizings();
			for (const Tuple &e : WMTKMesh::get_one_ring_edges_for_vertex(t))
				if (edge_sizings.at(e.eid(*this)) > 0.8)
					return false;
		}
		else
		{
			const std::unordered_map<size_t, double> edge_sizings = this->compute_edge_sizings();
			for (const Tuple &tet : WMTKMesh::get_one_ring_tets_for_vertex(t))
			{
				for (int i = 0; i < 6; ++i)
				{
					const Tuple e = WMTKMesh::tuple_from_edge(tet.tid(*this), i);
					if (edge_sizings.at(e.eid(*this)) > 0.8)
						return false;
				}
			}
		}

		return true;
	}

	template <typename WMTKMesh>
	template <typename Candidates>
	typename SizingFieldRemesher<WMTKMesh>::SparseSizingField
	SizingFieldRemesher<WMTKMesh>::compute_contact_sizing_field_from_candidates(
		const Candidates &candidates,
		const ipc::CollisionMesh &collision_mesh,
		const Eigen::MatrixXd &V,
		const double dhat) const
	{
		const Eigen::MatrixXd &V_rest = collision_mesh.rest_positions();
		const Eigen::MatrixXi &E = collision_mesh.edges();
		const Eigen::MatrixXi &F = collision_mesh.faces();

		Eigen::VectorXd min_distance_per_vertex = Eigen::VectorXd::Constant(V.rows(), dhat * dhat);

		SparseSizingField sizing_field;
		for (const auto &candidate : candidates)
		{
			const double distance_sqr = candidate.compute_distance(V, E, F);
			const double rest_distance_sqr = candidate.compute_distance(V_rest, E, F);
			const ipc::VectorMax12d distance_grad = candidate.compute_distance_gradient(V, E, F);
			const auto vertices = candidate.vertex_ids(E, F);
			for (int i = 0; i < 4; i++)
			{
				const long vi = vertices[i];
				if (vi < 0 || vi > V.rows() - this->obstacle().n_vertices())
					continue;
				if (distance_sqr / rest_distance_sqr >= 0.1 || distance_sqr > min_distance_per_vertex[vi])
					continue;

				VectorNd g = distance_grad.segment<Super::DIM>(Super::DIM * i) / (2 * sqrt(distance_sqr));

				sizing_field[collision_mesh.to_full_vertex_id(vi)] = g * (g.transpose() / distance_sqr);
				min_distance_per_vertex[vi] = distance_sqr;
			}
		}

		return sizing_field;
	}

	template <class WMTKMesh>
	typename SizingFieldRemesher<WMTKMesh>::SparseSizingField
	SizingFieldRemesher<WMTKMesh>::compute_contact_sizing_field() const
	{
		Eigen::MatrixXd V_rest = this->rest_positions();
		utils::append_rows(V_rest, this->obstacle().v());

		ipc::CollisionMesh collision_mesh = ipc::CollisionMesh::build_from_full_mesh(
			V_rest, this->boundary_edges(), this->boundary_faces());

		Eigen::MatrixXd V = this->positions();
		if (this->obstacle().n_vertices())
			utils::append_rows(V, this->obstacle().v() + this->obstacle_displacements());

		V_rest = collision_mesh.rest_positions();
		V = collision_mesh.vertices(V);
		const Eigen::MatrixXi &E = collision_mesh.edges();
		const Eigen::MatrixXi &F = collision_mesh.faces();

		const double dhat = state.args["contact"]["dhat"].template get<double>();

		ipc::HashGrid hash_grid;
		hash_grid.build(V, E, F, dhat / 2);
		if constexpr (Super::DIM == 2)
		{
			ipc::Candidates candidates;
			hash_grid.detect_edge_vertex_candidates(candidates.ev_candidates);
			return this->compute_contact_sizing_field_from_candidates(
				candidates.ev_candidates, collision_mesh, V, dhat);
		}
		else
		{
			ipc::Candidates candidates;
			hash_grid.detect_face_vertex_candidates(candidates.fv_candidates);
			// NOTE: ignoring edge-edge candidates for now
			// hash_grid.detect_edge_edge_candidates(candidates.ee_candidates);
			return this->compute_contact_sizing_field_from_candidates(
				candidates.fv_candidates, collision_mesh, V, dhat);
		}
	}

	template <class WMTKMesh>
	typename SizingFieldRemesher<WMTKMesh>::SparseSizingField
	SizingFieldRemesher<WMTKMesh>::smooth_contact_sizing_field(
		const SparseSizingField &sizing_field) const
	{
		SparseSizingField smoothed_sizing_field;
		for (const Tuple &f : this->boundary_facets())
		{
			const auto vids = this->facet_vids(f);
			MatrixNd M = MatrixNd::Zero();
			for (const size_t vid : vids)
				if (sizing_field.find(vid) != sizing_field.end())
					M += sizing_field.at(vid);
			M /= vids.size(); // average

			std::vector<Tuple> edges;
			if constexpr (Super::DIM == 2)
				edges.push_back(f);
			else
				edges = {{f, f.switch_edge(*this), f.switch_vertex(*this).switch_edge(*this)}};

			for (const Tuple &e : edges)
			{
				if (sizing_field.find(e.eid(*this)) == sizing_field.end())
					smoothed_sizing_field[e.eid(*this)] = MatrixNd::Zero();
				smoothed_sizing_field[e.eid(*this)] += M / edges.size();
			}
		}

		return smoothed_sizing_field;
	}

	template <class WMTKMesh>
	std::unordered_map<size_t, double>
	SizingFieldRemesher<WMTKMesh>::compute_edge_sizings() const
	{
		const SparseSizingField sizing_field =
			combine_sizing_fields(
				compute_elasticity_sizing_field(),
				smooth_contact_sizing_field(compute_contact_sizing_field()));

		std::unordered_map<size_t, double> edge_sizings;
		for (const Tuple &e : WMTKMesh::get_edges())
		{
			const std::array<size_t, 2> vids =
				{{e.vid(*this), e.switch_vertex(*this).vid(*this)}};

			const VectorNd xi_bar = vertex_attrs[vids[0]].rest_position;
			const VectorNd xj_bar = vertex_attrs[vids[1]].rest_position;

			const VectorNd xij_bar = xj_bar - xi_bar;

			const MatrixNd M = sizing_field.at(e.eid(*this));

			edge_sizings[e.eid(*this)] = sqrt(xij_bar.transpose() * M * xij_bar);
		}
		return edge_sizings;
	}

	template <class WMTKMesh>
	typename SizingFieldRemesher<WMTKMesh>::SparseSizingField
	SizingFieldRemesher<WMTKMesh>::compute_elasticity_sizing_field() const
	{
		std::unordered_map<size_t, MatrixNd> Fs, Fs_inv;
		for (const Tuple &t : this->get_elements())
		{
			const auto vids = this->element_vids(t);
			MatrixNd Dm, Ds;
			if constexpr (std::is_same_v<WMTKMesh, wmtk::TriMesh>)
			{
				Dm.col(0) = vertex_attrs[vids[1]].rest_position - vertex_attrs[vids[0]].rest_position;
				Dm.col(1) = vertex_attrs[vids[2]].rest_position - vertex_attrs[vids[0]].rest_position;
				Ds.col(0) = vertex_attrs[vids[1]].position - vertex_attrs[vids[0]].position;
				Ds.col(1) = vertex_attrs[vids[2]].position - vertex_attrs[vids[0]].position;
			}
			else
			{
				Dm.col(0) = vertex_attrs[vids[1]].rest_position - vertex_attrs[vids[0]].rest_position;
				Dm.col(1) = vertex_attrs[vids[2]].rest_position - vertex_attrs[vids[0]].rest_position;
				Dm.col(2) = vertex_attrs[vids[3]].rest_position - vertex_attrs[vids[0]].rest_position;
				Ds.col(0) = vertex_attrs[vids[1]].position - vertex_attrs[vids[0]].position;
				Ds.col(1) = vertex_attrs[vids[2]].position - vertex_attrs[vids[0]].position;
				Ds.col(2) = vertex_attrs[vids[3]].position - vertex_attrs[vids[0]].position;
			}
			Fs[this->element_id(t)] = Ds * Dm.inverse();
			Fs_inv[this->element_id(t)] = Dm * Ds.inverse();
		}

		const double k = 1.0;

		std::unordered_map<size_t, VectorNd> element_centers;
		for (const Tuple &t : this->get_elements())
		{
			VectorNd center = VectorNd::Zero();
			const auto &vids = this->element_vids(t);
			for (const size_t &vid : vids)
				center += vertex_attrs[vid].rest_position;
			element_centers[this->element_id(t)] = center / vids.size();
		}

		SparseSizingField element_sizing_field;
		for (const Tuple &t : this->get_elements())
		{
			const size_t tid = this->element_id(t);

			// std::unordered_set<size_t> adjacent_tids;
			// for (const Tuple &v : this->element_vertices(t))
			// 	for (const Tuple &adj_t : this->get_one_ring_elements_for_vertex(v))
			// 		adjacent_tids.insert(this->element_id(adj_t));
			// adjacent_tids.erase(tid);

			// const VectorNd &t_center = element_centers.at(tid);

			// MatrixNd M = MatrixNd::Zero();
			// for (const size_t adj_tid : adjacent_tids)
			// {
			// 	const VectorNd dt = element_centers.at(adj_tid) - t_center;
			// 	M += dt * (dt.transpose() / dt.squaredNorm())
			// 		 * std::max((Fs[tid] * Fs_inv[adj_tid]).norm(),
			// 					(Fs[adj_tid] * Fs_inv[tid]).norm())
			// 		 / dt.norm();
			// }
			// M.array() *= k / adjacent_tids.size();

			element_sizing_field[tid] = Fs[tid].transpose() * Fs[tid];
		}
		assert(!element_sizing_field.empty());

		SparseSizingField sizing_field;
		for (const Tuple &e : this->get_edges())
		{
			const size_t eid = e.eid(*this);
			const auto &incident_elements = this->get_incident_elements_for_edge(e);
			for (const Tuple &t : incident_elements)
			{
				if (sizing_field.find(eid) == sizing_field.end())
					sizing_field[eid] = MatrixNd::Zero();
				sizing_field[eid] += element_sizing_field[this->element_id(t)];
			}
			sizing_field[eid] /= incident_elements.size() * std::pow(state.starting_max_edge_length, 2);
		}

		return sizing_field;
	}

	template <class WMTKMesh>
	typename SizingFieldRemesher<WMTKMesh>::SparseSizingField
	SizingFieldRemesher<WMTKMesh>::combine_sizing_fields(
		const SparseSizingField &field1,
		const SparseSizingField &field2)
	{
		SparseSizingField field = field1;
		for (const auto &[eid, M] : field2)
		{
			if (field.find(eid) == field.end())
				field[eid] = M;
			else
				field[eid] = (field[eid] + M) / 2;
		}
		return field;
	}

	// -------------------------------------------------------------------------
	// Template specializations

	template class SizingFieldRemesher<wmtk::TriMesh>;
	template class SizingFieldRemesher<wmtk::TetMesh>;
} // namespace polyfem::mesh

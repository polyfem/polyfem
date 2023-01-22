#include "SizingFieldRemesher.hpp"

#include <ipc/collision_mesh.hpp>
#include <ipc/broad_phase/hash_grid.hpp>

namespace polyfem::mesh
{
	// Edge splitting
	template <class WMTKMesh>
	void SizingFieldRemesher<WMTKMesh>::split_edges()
	{
		Operations splits;
		const std::unordered_map<size_t, double> edge_sizings =
			this->compute_contact_edge_sizings();
		for (const Tuple &e : WMTKMesh::get_edges())
			if (edge_sizings.at(e.eid(*this)) <= 1)
				splits.emplace_back("edge_split", e);

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
		const std::unordered_map<size_t, double> edge_sizings =
			this->compute_contact_edge_sizings();
		for (const Tuple &e : WMTKMesh::get_edges())
			if (edge_sizings.at(e.eid(*this)) >= 0.8)
				collapses.emplace_back("edge_collapse", e);

		if (collapses.empty())
			return;

		executor.priority = [&](const WildRemesher<WMTKMesh> &m, std::string op, const Tuple &t) -> double {
			return edge_sizings.at(t.eid(m));
		};

		executor(*this, collapses);
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
		const Eigen::MatrixXd &V_rest = collision_mesh.vertices_at_rest();
		const Eigen::MatrixXi &E = collision_mesh.edges();
		const Eigen::MatrixXi &F = collision_mesh.faces();

		Eigen::VectorXd min_distance_per_vertex = Eigen::VectorXd::Constant(V.rows(), dhat * dhat);

		SparseSizingField sizing_field;
		for (const auto &candidate : candidates)
		{
			const long vi = candidate.vertex_index;
			if (vi > V.rows() - this->obstacle().n_vertices())
				continue;
			const double distance_sqr = candidate.compute_distance(V, E, F);
			const double rest_distance_sqr = candidate.compute_distance(V_rest, E, F);
			if (distance_sqr / rest_distance_sqr < 0.1
				&& distance_sqr < min_distance_per_vertex[vi])
			{
				ipc::VectorMax12d distance_grad = candidate.compute_distance_gradient(V, E, F);
				distance_grad = distance_grad.head(V.cols()) / (2 * sqrt(distance_sqr));

				sizing_field[collision_mesh.to_full_vertex_id(vi)] =
					distance_grad * (distance_grad.transpose() / distance_sqr);
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

		V_rest = collision_mesh.vertices_at_rest();
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
			M /= vids.size(); // smooth (re-distibute)
			for (const size_t vid : vids)
			{
				if (smoothed_sizing_field.find(vid) == smoothed_sizing_field.end())
					smoothed_sizing_field[vid] = MatrixNd::Zero();
				smoothed_sizing_field[vid] += M;
			}
		}

		return smoothed_sizing_field;
	}

	template <class WMTKMesh>
	std::unordered_map<size_t, double>
	SizingFieldRemesher<WMTKMesh>::compute_contact_edge_sizings() const
	{
		const SparseSizingField sizing_field =
			smooth_contact_sizing_field(compute_contact_sizing_field());

		std::unordered_map<size_t, double> edge_sizings;
		for (const Tuple &e : WMTKMesh::get_edges())
		{
			const std::array<size_t, 2> vids =
				{{e.vid(*this), e.switch_vertex(*this).vid(*this)}};

			const VectorNd xi_bar = vertex_attrs[vids[0]].rest_position;
			const VectorNd xj_bar = vertex_attrs[vids[1]].rest_position;

			const VectorNd xij_bar = xj_bar - xi_bar;

			auto iter = sizing_field.find(vids[0]);
			const MatrixNd Mi = iter != sizing_field.end() ? iter->second : MatrixNd::Zero();

			iter = sizing_field.find(vids[1]);
			const MatrixNd Mj = iter != sizing_field.end() ? iter->second : MatrixNd::Zero();

			const MatrixNd M = (Mi + Mj) / 2;

			edge_sizings[e.eid(*this)] = xij_bar.transpose() * M * xij_bar;
		}
		return edge_sizings;
	}

	// -------------------------------------------------------------------------
	// Template specializations

	template class SizingFieldRemesher<wmtk::TriMesh>;
	template class SizingFieldRemesher<wmtk::TetMesh>;
} // namespace polyfem::mesh

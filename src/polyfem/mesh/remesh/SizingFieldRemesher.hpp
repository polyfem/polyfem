#pragma once

#include <polyfem/mesh/remesh/WildRemesher.hpp>

namespace polyfem::mesh
{
	template <class WMTKMesh>
	class SizingFieldRemesher : public WildRemesher<WMTKMesh>
	{
	private:
		using Super = WildRemesher<WMTKMesh>;
		using This = SizingFieldRemesher<WMTKMesh>;

	protected:
		using Super::args;
		using Super::executor;
		using Super::state;

	public:
		using Super::vertex_attrs;
		using Tuple = typename Super::Tuple;
		using Operations = typename Super::Operations;
		using VectorNd = typename Super::VectorNd;
		using MatrixNd = Eigen::Matrix<double, Super::DIM, Super::DIM>;

		SizingFieldRemesher(
			const State &state,
			const Eigen::MatrixXd &obstacle_displacements,
			const Eigen::MatrixXd &obstacle_vals,
			const double current_time,
			const double starting_energy)
			: Super(state, obstacle_displacements, obstacle_vals, current_time, starting_energy)
		{
		}

		virtual ~SizingFieldRemesher() {};

		// Edge splitting
		void split_edges() override;
		bool split_edge_before(const Tuple &t) override;

		// Edge collapse
		void collapse_edges() override;
		bool collapse_edge_after(const Tuple &t) override;

		using SparseSizingField = std::unordered_map<size_t, MatrixNd>;

		SparseSizingField compute_contact_sizing_field() const;
		SparseSizingField smooth_contact_sizing_field(
			const SparseSizingField &sizing_field) const;

		SparseSizingField compute_elasticity_sizing_field() const;

		std::unordered_map<size_t, double> compute_edge_sizings() const;

		static SparseSizingField combine_sizing_fields(
			const SparseSizingField &field1,
			const SparseSizingField &field2);

	private:
		template <typename Candidates>
		SparseSizingField compute_contact_sizing_field_from_candidates(
			const Candidates &candidates,
			const ipc::CollisionMesh &collision_mesh,
			const Eigen::MatrixXd &V,
			const double dhat) const;
	};

	using SizingFieldTriRemesher = SizingFieldRemesher<wmtk::TriMesh>;
	using SizingFieldTetRemesher = SizingFieldRemesher<wmtk::TetMesh>;

} // namespace polyfem::mesh

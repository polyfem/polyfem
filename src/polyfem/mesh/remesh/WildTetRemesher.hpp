#pragma once

#include <polyfem/mesh/remesh/WildRemesher.hpp>
#include <polyfem/mesh/remesh/wild_remesh/OperationCache.hpp>

#include <wmtk/TetMesh.h>

namespace polyfem::mesh
{
	class WildTetRemesher : public WildRemesher<wmtk::TetMesh>
	{
		// --------------------------------------------------------------------
		// typedefs
	private:
		using Super = WildRemesher<wmtk::TetMesh>;

		// --------------------------------------------------------------------
		// constructors
	public:
		/// @brief Construct a new WildTetRemesher object
		/// @param state Simulation current state
		WildTetRemesher(
			const State &state,
			const Eigen::MatrixXd &obstacle_displacements,
			const Eigen::MatrixXd &obstacle_vals,
			const double current_time,
			const double starting_energy);

		virtual ~WildTetRemesher(){};

	protected:
		/// @brief Create an internal mesh representation and associate attributes
		void init_attributes_and_connectivity(
			const size_t num_vertices,
			const Eigen::MatrixXi &elements) override;

		// --------------------------------------------------------------------
		// main functions
	public:
		/// @brief Execute the remeshing
		/// @param split Perform splitting operations
		/// @param collapse Perform collapsing operations
		/// @param smooth Perform smoothing operations
		/// @param swap Perform edge swapping operations
		/// @param max_ops Maximum number of operations to perform (default: unlimited)
		/// @return True if any operation was performed.
		bool execute(
			const bool split = true,
			const bool collapse = false,
			const bool smooth = false,
			const bool swap = false,
			const double max_ops_percent = -1) override { throw std::runtime_error("Not implemented!"); }

	protected:
		// Smoothing
		// bool smooth_before(const Tuple &t) override;
		// bool smooth_after(const Tuple &t) override;

		// Edge splitting
		bool split_edge_after(const Tuple &t) override;

		// Edge collapse
		// bool collapse_edge_before(const Tuple &t) override;
		// bool collapse_edge_after(const Tuple &t) override;

		// 3-2 Edge swap
		// bool swap_edge_before(const Tuple &t) override;
		// bool swap_edge_after(const Tuple &t) override;

		// 4-4 Edge swap
		// bool swap_edge_44_before(const Tuple &t) override;
		// bool swap_edge_44_after(const Tuple &t) override;

		// 2-3 Face swap
		// bool swap_face_before(const Tuple &t) override;
		// bool swap_face_after(const Tuple &t) override;

		/// @brief Check if a tetrahedron is inverted
		bool is_inverted(const Tuple &loc) const override;

		// --------------------------------------------------------------------
		// getters

		std::vector<Tuple> get_elements() const override { return get_tets(); }

		// --------------------------------------------------------------------
		// setters

		// --------------------------------------------------------------------
		// utilities
	public:
		/// @brief Compute the volume of a tetrahedron element.
		double element_volume(const Tuple &e) const override;

		/// @brief Is the given tuple on the boundary of the mesh?
		bool is_on_boundary(const Tuple &t) const override
		{
			return !t.switch_tetrahedron(*this).has_value();
		}

		/// @brief Get the boundary facets of the mesh
		std::vector<Tuple> boundary_facets() const override;

		/// @brief Get the vertex ids of a boundary facet.
		std::array<size_t, 3> boundary_facet_vids(const Tuple &t) const override
		{
			return {{
				t.vid(*this),
				t.switch_vertex(*this).vid(*this),
				t.switch_edge(*this).switch_vertex(*this).vid(*this),
			}};
		}

		/// @brief Get the vertex ids of an element.
		std::array<Tuple, 4> element_vertices(const Tuple &t) const override
		{
			return oriented_tet_vertices(t);
		}

		/// @brief Get the vertex ids of an element.
		std::array<size_t, 4> element_vids(const Tuple &t) const override
		{
			return oriented_tet_vids(t);
		}

		/// @brief Get the one ring of elements around a vertex.
		std::vector<Tuple> get_one_ring_elements_for_vertex(const Tuple &t) const override
		{
			return get_one_ring_tets_for_vertex(t);
		}

		size_t facet_id(const Tuple &t) const { return t.fid(*this); }
		size_t element_id(const Tuple &t) const { return t.tid(*this); }

		Tuple tuple_from_facet(size_t elem_id, int local_facet_id) const
		{
			return tuple_from_face(elem_id, local_facet_id);
		}

	protected:
		void cache_split_edge(const Tuple &e) override
		{
			op_cache = TetOperationCache::split_edge(*this, e);
		}

		void map_edge_split_boundary_attributes(
			const Tuple &new_vertex,
			const FaceMap<BoundaryAttributes> &old_faces,
			const size_t old_v0_id,
			const size_t old_v1_id);

		void map_edge_split_element_attributes(
			const Tuple &new_vertex,
			const TetMap<ElementAttributes> &old_elements,
			const size_t old_v0_id,
			const size_t old_v1_id);

		/// @brief Compute the average elastic energy of the faces containing an edge.
		// double edge_elastic_energy(const Tuple &e) const;

		/// @brief Create a vector of all the new edge after an operation.
		/// @param tris New triangles.
		// std::vector<Tuple> new_edges_after(const std::vector<Tuple> &tets) const;

		// --------------------------------------------------------------------
		// parameters

		// --------------------------------------------------------------------
		// members
	protected:
		// TODO: make this thread local
		TetOperationCache op_cache;
	};

} // namespace polyfem::mesh

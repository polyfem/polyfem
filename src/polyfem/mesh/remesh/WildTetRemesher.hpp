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
	protected:
		// Smoothing
		// bool smooth_before(const Tuple &t) override;
		// bool smooth_after(const Tuple &t) override;

		// Edge splitting
		bool split_edge_after(const Tuple &t) override;

		// Edge collapse

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

		/// @brief Exports boundary edges of the stored mesh
		Eigen::MatrixXi boundary_edges() const override;
		/// @brief Exports boundary faces of the stored mesh
		Eigen::MatrixXi boundary_faces() const override;

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

		/// @brief Get the id of a facet (edge for triangle, triangle for tetrahedra)
		size_t facet_id(const Tuple &t) const { return t.fid(*this); }

		/// @brief Get the id of an element (triangle or tetrahedra)
		size_t element_id(const Tuple &t) const { return t.tid(*this); }

		/// @brief Get a tuple of a element with a local facet
		Tuple tuple_from_facet(size_t elem_id, int local_facet_id) const
		{
			return tuple_from_face(elem_id, local_facet_id);
		}

		/// @brief Get the incident elements for an edge
		std::vector<Tuple> get_incident_elements_for_edge(const Tuple &t) const override
		{
			return get_incident_tets_for_edge(t);
		}

		std::vector<Tuple> get_one_ring_boundary_edges_for_vertex(const Tuple &v) const;
		std::vector<Tuple> get_one_ring_boundary_faces_for_vertex(const Tuple &v) const;
		std::array<Tuple, 2> get_boundary_faces_for_edge(const Tuple &e) const;
		Tuple opposite_vertex_on_face(const Tuple &e) const
		{
			return e.switch_edge(*this).switch_vertex(*this);
		}

		bool is_edge_on_boundary(const Tuple &e) const override;
		bool is_edge_on_body_boundary(const Tuple &e) const override;
		bool is_vertex_on_boundary(const Tuple &v) const override { throw std::runtime_error("Not implemented"); }
		bool is_vertex_on_body_boundary(const Tuple &v) const override { throw std::runtime_error("Not implemented"); }

		CollapseEdgeTo collapse_boundary_edge_to(const Tuple &e) const override;

	protected:
		double local_energy() const override { return op_cache.local_energy; }

		// edge split

		/// @brief Cache the split edge operation
		/// @param e edge tuple
		/// @param local_energy local energy
		void cache_split_edge(const Tuple &e, const double local_energy) override
		{
			op_cache = TetOperationCache::split_edge(*this, e);
			op_cache.local_energy = local_energy;
		}

		void map_edge_split_edge_attributes(
			const Tuple &new_vertex,
			const EdgeMap<EdgeAttributes> &old_edges,
			const size_t old_v0_id,
			const size_t old_v1_id);

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

		// edge collapse

		/// @brief Cache the edge collapse operation
		/// @param e edge tuple
		/// @param local_energy local energy
		/// @param collapse_to collapse to which vertex
		void cache_collapse_edge(const Tuple &e, const double local_energy, const CollapseEdgeTo collapse_to) override
		{
			op_cache = TetOperationCache::collapse_edge(*this, e);
			op_cache.local_energy = local_energy;
			op_cache.collapse_to = collapse_to;
		}

		/// @brief Map the vertex attributes for edge collapse
		/// @param t new vertex tuple
		void map_edge_collapse_vertex_attributes(const Tuple &t) override;

		/// @brief Map the edge attributes for edge collapse
		/// @param t new vertex tuple
		void map_edge_collapse_edge_attributes(const Tuple &t);

		/// @brief Map the boundary attributes for edge collapse
		/// @param t new vertex tuple
		void map_edge_collapse_boundary_attributes(const Tuple &t) override;

		// No need to map boundary attributes for edge collapse because no new boundary is created
		// No need to map element attributes for edge collapse because no new element is created

		// --------------------------------------------------------------------
		// parameters

		// --------------------------------------------------------------------
		// members
	public:
		/// @brief Get a reference to an edge's attributes
		/// @param e_id edge id
		/// @return reference to the edge's attributes
		EdgeAttributes &edge_attr(const size_t e_id) override { return edge_attrs[e_id]; }

		/// @brief Get a const reference to an edge's attributes
		/// @param e_id edge id
		/// @return const reference to the edge's attributes
		const EdgeAttributes &edge_attr(const size_t e_id) const override { return edge_attrs[e_id]; }

		wmtk::AttributeCollection<EdgeAttributes> edge_attrs;

	protected:
		// TODO: make this thread local
		TetOperationCache op_cache;
	};

} // namespace polyfem::mesh

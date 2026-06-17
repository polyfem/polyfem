#pragma once

#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/basis/InterfaceData.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/mesh/MeshNodes.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace polyfem::varform
{
	class FESpace;

	class GeometryMapping
	{
	public:
		/// Number of globally indexed scalar geometry basis functions.
		int n_bases = 0;

		/// Per-element scalar bases used to interpolate physical coordinates.
		std::shared_ptr<std::vector<basis::ElementBases>> bases;

		/// Polynomial degree of the geometry mapping on each mesh element.
		Eigen::VectorXi disc_orders;

		/// Physical boundary samples for 2D polygonal elements.
		std::map<int, Eigen::MatrixXd> polys;

		/// Physical vertices and face connectivity for 3D polyhedral elements.
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

		/// Optional primitive-to-node mapping for the geometry bases.
		std::shared_ptr<mesh::MeshNodes> mesh_nodes;

		void init_from_fe_space(const FESpace &space);

		void reset()
		{
			n_bases = 0;
			bases = nullptr;
			disc_orders.resize(0);
			polys.clear();
			polys_3d.clear();
			mesh_nodes = nullptr;
		}
	};

	/// A finite-element space for one scalar- or vector-valued field.
	class FESpace
	{
	public:
		/// Number of field components per scalar basis function.
		int value_dim = 1;

		/// Number of globally indexed scalar basis functions in the space.
		int n_bases = 0;

		/// Per-element basis data.
		std::shared_ptr<std::vector<basis::ElementBases>> bases;

		/// Primary polynomial degree for each mesh element.
		Eigen::VectorXi disc_orders;

		/// Secondary polynomial degree for anisotropic bases, e.g. prisms.
		Eigen::VectorXi disc_ordersq;

		/// Polygonal-basis construction data, indexed by element ID.
		std::map<int, basis::InterfaceData> poly_edge_to_data;

		/// Physical boundary samples for 2D polygonal elements.
		std::map<int, Eigen::MatrixXd> polys;

		/// Physical vertices and face connectivity for 3D polyhedral elements.
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

		/// Optional primitive-to-node mapping for this FE space.
		std::shared_ptr<mesh::MeshNodes> mesh_nodes;

		/// Geometric mapping used to integrate this FE space.
		std::shared_ptr<GeometryMapping> geometry;

		Eigen::VectorXi space_in_node_to_node;
		Eigen::VectorXi space_in_primitive_to_primitive;

		int ndof() const
		{
			return n_bases * value_dim;
		}

		bool has_bases() const
		{
			return bases != nullptr;
		}

		bool is_iso_parametric() const
		{
			return geometry && geometry->bases == bases;
		}

		const std::vector<basis::ElementBases> &basis_list() const
		{
			assert(bases);
			return *bases;
		}

		const std::vector<basis::ElementBases> &geometry_basis_list() const
		{
			assert(geometry);
			assert(geometry->bases);
			return *geometry->bases;
		}

		void reset()
		{
			value_dim = 1;
			n_bases = 0;
			bases = nullptr;
			disc_orders.resize(0);
			disc_ordersq.resize(0);
			poly_edge_to_data.clear();
			polys.clear();
			polys_3d.clear();
			mesh_nodes = nullptr;
			geometry = nullptr;

			space_in_node_to_node.resize(0);
			space_in_primitive_to_primitive.resize(0);
		}
	};

	inline void GeometryMapping::init_from_fe_space(const FESpace &space)
	{
		n_bases = space.n_bases;
		bases = space.bases;
		disc_orders = space.disc_orders;
		polys = space.polys;
		polys_3d = space.polys_3d;
		mesh_nodes = space.mesh_nodes;
	}

	/// Temporary compatibility wrapper for boundary data belonging to one FE space.
	struct VarFormBoundaryState
	{
		std::vector<int> boundary_nodes;
		std::vector<mesh::LocalBoundary> total_local_boundary;
		std::vector<mesh::LocalBoundary> local_boundary;
		std::vector<mesh::LocalBoundary> local_neumann_boundary;
		std::vector<mesh::LocalBoundary> local_pressure_boundary;
		std::unordered_map<int, std::vector<mesh::LocalBoundary>> local_pressure_cavity;
		std::vector<int> pressure_boundary_nodes;
		std::vector<int> dirichlet_nodes;
		std::vector<RowVectorNd> dirichlet_nodes_position;
		std::vector<int> neumann_nodes;
		std::vector<RowVectorNd> neumann_nodes_position;

		void clear_boundary_conditions()
		{
			boundary_nodes.clear();
			local_boundary.clear();
			local_neumann_boundary.clear();
			local_pressure_boundary.clear();
			local_pressure_cavity.clear();
			pressure_boundary_nodes.clear();
			dirichlet_nodes.clear();
			dirichlet_nodes_position.clear();
			neumann_nodes.clear();
			neumann_nodes_position.clear();
		}

		void normalize_boundary_nodes()
		{
			std::sort(boundary_nodes.begin(), boundary_nodes.end());
			boundary_nodes.erase(std::unique(boundary_nodes.begin(), boundary_nodes.end()), boundary_nodes.end());
		}

		void reset()
		{
			total_local_boundary.clear();
			clear_boundary_conditions();
		}
	};
} // namespace polyfem::varform

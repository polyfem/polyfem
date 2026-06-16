#pragma once

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/MeshNodes.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>

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

		// DESIGN NOTE:
		// In multi-physics setting, the old iso_parametric flag is ambiguous. Thus instead of a flag we store bases as a shared ptr. Then user can evaluate:
		// bool iso_parametric = (target finite element bases == geometry bases);

		/// Per-element scalar bases used to interpolate physical coordinates.
		std::shared_ptr<std::vector<basis::ElementBases>> bases;

		/// Polynomial degree of the geometry mapping on each mesh element.
		Eigen::VectorXi disc_orders;

		/// Physical boundary samples for two-dimensional polygonal elements,
		/// indexed by element ID. Currently for output IO only.
		std::map<int, Eigen::MatrixXd> polys;

		/// Physical vertices and face connectivity for three-dimensional polyhedral
		/// elements, indexed by element ID. Currently for output IO only.
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

		/// Optional mapping between mesh primitives and geometry-basis nodes. It is
		/// nullptr when basis does not provide a MeshNodes representation,
		/// as is currently the case for spline bases.
		std::shared_ptr<mesh::MeshNodes> mesh_nodes;

		/// Initialize from an existing FE space. This is a QOL helper for iso-parametric case.
		void init_from_fe_space(const FESpace &space);
	};

	/// @brief A finite-element space for one scalar- or vector-valued field.
	class FESpace
	{
	public:
		/// Number of field components associated with each scalar basis function.
		/// This is 1 for a scalar field and typically the mesh dimension for a
		/// displacement or velocity field.
		int value_dim = 1;

		/// Number of globally indexed scalar basis functions in the space.
		int n_bases = 0;

		/// Per-element basis info.
		std::shared_ptr<std::vector<basis::ElementBases>> bases;

		/// Primary polynomial degree for each mesh element.
		Eigen::VectorXi disc_orders;

		/// Secondary polynomial degree for for each mesh element for aniostropic basis.
		/// (ie. prism). Ignored by isotropic basis.
		Eigen::VectorXi disc_ordersq;

		/// Physical boundary samples for two-dimensional polygonal elements,
		/// indexed by element ID. Currently for output IO only.
		std::map<int, Eigen::MatrixXd> polys;

		/// Physical vertices and face connectivity for three-dimensional polyhedral
		/// elements, indexed by element ID. Currently for output IO only.
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polys_3d;

		/// Optional mapping between mesh primitives and geometry-basis nodes. It is
		/// nullptr when basis does not provide a MeshNodes representation,
		/// as is currently the case for spline bases.
		std::shared_ptr<mesh::MeshNodes> mesh_nodes;

		// Geometric mapping for weak form integration. This can be different from FE space
		// basis as we sometimes want low order geometric basis but would prefer higher order basis for
		// FE fields.
		std::shared_ptr<GeometryMapping> geometry;

		// DESIGN NOTE:
		// geometry mapping is stored as shared ptr because:
		// 1. In future we might want to integrate on different geometry discretizetion.
		// 2. In most cases, we probably just want a simple shared geometry mapping.

		/// @brief Return the total number of degrees of freedom in this space.
		int ndof() const
		{
			return n_bases * value_dim;
		}
	};

	/// THIS IS A TEMPORARY COMPATIBILITY WRAPPER!!!!
	/// It only usage is to bundle some data. It does not solve any architectural issue
	/// hence should be refactor later. Its just a temp fix. safe to skip.
	struct VarFormBoundaryState
	{
		/// Global field DOF indices constrained by essential boundary conditions.
		std::vector<int> boundary_nodes;

		/// Complete local boundary generated while constructing the field basis,
		/// before boundary-condition classification removes or separates entries.
		std::vector<mesh::LocalBoundary> total_local_boundary;

		/// Local boundary entries used for the primary field boundary conditions.
		std::vector<mesh::LocalBoundary> local_boundary;

		/// Local boundary entries carrying Neumann conditions.
		std::vector<mesh::LocalBoundary> local_neumann_boundary;

		/// Local boundary entries carrying pressure or normal-traction conditions.
		std::vector<mesh::LocalBoundary> local_pressure_boundary;

		/// Pressure boundary entries grouped by cavity identifier.
		std::unordered_map<int, std::vector<mesh::LocalBoundary>> local_pressure_cavity;

		/// Global field DOF indices participating in pressure boundary conditions.
		std::vector<int> pressure_boundary_nodes;

		/// Global basis indices with prescribed Dirichlet values.
		std::vector<int> dirichlet_nodes;

		/// Physical positions corresponding one-to-one with dirichlet_nodes.
		std::vector<RowVectorNd> dirichlet_nodes_position;

		/// Global basis indices associated with nodal Neumann data.
		std::vector<int> neumann_nodes;

		/// Physical positions corresponding one-to-one with neumann_nodes.
		std::vector<RowVectorNd> neumann_nodes_position;
	};

} // namespace polyfem::varform
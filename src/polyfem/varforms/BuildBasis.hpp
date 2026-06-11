#pragma once

#include <polyfem/Common.hpp>

#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/mesh/MeshNodes.hpp>

#include <Eigen/Dense>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace polyfem
{
	namespace assembler
	{
		class LinearAssembler;
	}

	namespace mesh
	{
		class Mesh;
	}
} // namespace polyfem

namespace polyfem::varform
{
	struct BasisBuildResult
	{
		/// Per-element basis functions.
		std::vector<basis::ElementBases> bases;
		/// Number of global basis functions.
		int n_bases = 0;
		/// Optional mapping between mesh primitives and geometry-basis nodes. It is
		/// nullptr when basis does not provide a MeshNodes representation,
		/// as is currently the case for spline bases.
		std::shared_ptr<mesh::MeshNodes> mesh_nodes;
		/// Mesh-boundary primitives and their local basis indices.
		std::vector<mesh::LocalBoundary> local_boundary;
		/// Mapped boundaries of two-dimensional polygonal elements.
		std::map<int, Eigen::MatrixXd> polygons;
		/// Mapped boundaries of three-dimensional polyhedral elements.
		std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXi>> polyhedra;
		/// Time spent constructing non-polygonal bases, in seconds.
		double standard_basis_time = 0;
		/// Time spent constructing polygonal or polyhedral bases, in seconds.
		double polygon_basis_time = 0;
	};

	/// @brief Build one finite-element basis.
	///
	/// @param mesh Input computational mesh.
	/// @param args Full input configuration containing basis and quadrature options.
	/// @param formulation Formulation name (Assembler name) used to select automatic quadrature orders.
	/// @param orders Per-element basis orders.
	/// @param orders_q Optional secondary order for anisomorphic basis (ie. prism).
	/// @param value_dim Number of components represented by the basis.
	/// @param is_geometry_basis Whether to build a geometry-mapping basis.
	/// @param linear_assembler Governing linear assembler used by MFS harmonic bases.
	/// It may be nullptr for non-RBF bases but is required when MFSHarmonic is selected.
	///
	/// @throws std::runtime_error If MFSHarmonic is selected without a linear assembler,
	/// or when an unsupported polygon basis is requested.
	BasisBuildResult build_basis(
		const mesh::Mesh &mesh,
		const json &args,
		const std::string &formulation,
		const Eigen::VectorXi &orders,
		const Eigen::VectorXi &orders_q,
		const int value_dim,
		const bool is_geometry_basis,
		const assembler::LinearAssembler *linear_assembler = nullptr);
} // namespace polyfem::varform

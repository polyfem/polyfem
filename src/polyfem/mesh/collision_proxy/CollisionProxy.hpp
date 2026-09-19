#pragma once

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>

#include <array>
#include <set>
#include <vector>

namespace polyfem::mesh
{
	enum class CollisionProxyTessellation
	{
		REGULAR,  ///< @brief Regular tessellation of the mesh
		IRREGULAR ///< @brief Irregular tessellation of the mesh (requires POLYFEM_WITH_TRIANGLE)
	};

	NLOHMANN_JSON_SERIALIZE_ENUM(
		CollisionProxyTessellation,
		{{CollisionProxyTessellation::REGULAR, "regular"},
		 {CollisionProxyTessellation::IRREGULAR, "irregular"}});

	/// @brief A collision proxy held in memory: exactly what its three files hold (the proxy mesh
	/// file, the weights hdf5 file, and the body-id text file). The in-memory overloads of
	/// load_collision_proxy and load_collision_proxy_collision_body_ids share all their code after
	/// the reading with the file overloads, so a proxy here and in files give the same collision mesh.
	struct CollisionProxyData
	{
		/// @brief The proxy mesh as read_surface_mesh returns it for the file: #V x 2 or #V x 3
		/// positions (a third column that is all zero is dropped, as for a file), codimensional
		/// vertices, codimensional edges, and triangles. A 2D proxy has no triangles; its edges are
		/// codim_edges. The codimensional vertices are not stored in the file: read_surface_mesh
		/// derives them as every vertex that no codim edge and no face uses, in increasing order,
		/// and the caller fills this field by the same rule.
		Eigen::MatrixXd vertices;
		Eigen::VectorXi codim_vertices;
		Eigen::MatrixXi codim_edges;
		Eigen::MatrixXi faces;

		/// @brief The displacement map as the weights file stores it (group "weight_triplets"):
		/// entry k has weight weight_values[k] from FE input node weight_cols[k] to proxy vertex
		/// weight_rows[k]; weight_shape is [#proxy vertices, #FE input nodes].
		Eigen::VectorXd weight_values;
		Eigen::VectorXi weight_rows;
		Eigen::VectorXi weight_cols;
		std::array<long, 2> weight_shape = {{0, 0}};

		/// @brief Collision body ids as the body-id file stores them: one list per primitive
		/// (triangle, or edge when the proxy has no triangles), in primitive order. Empty means no
		/// body ids, like an absent file.
		std::vector<std::vector<int>> collision_body_ids;
	};

	/// @brief Build a collision proxy mesh by upsampling a given mesh.
	/// @param[in] bases Bases for elements
	/// @param[in] geom_bases Geometry bases for elements
	/// @param[in] total_local_boundary Local boundaries for elements
	/// @param[in] n_bases Number of bases (nodes)
	/// @param[in] dim Dimension of the mesh
	/// @param[in] max_edge_length Maximum edge length of the proxy mesh
	/// @param[out] proxy_vertices Output vertices of the proxy mesh
	/// @param[out] proxy_faces Output faces of the proxy mesh
	/// @param[out] displacement_map Output displacement map from proxy mesh to original mesh
	/// @param[in] tessellation Type of tessellation to use
	void build_collision_proxy(
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		const int n_bases,
		const int dim,
		const double max_edge_length,
		Eigen::MatrixXd &proxy_vertices,
		Eigen::MatrixXi &proxy_faces,
		std::vector<Eigen::Triplet<double>> &displacement_map,
		const CollisionProxyTessellation tessellation = CollisionProxyTessellation::REGULAR);

	/// @brief Build a collision proxy displacement map for a given mesh and proxy mesh.
	/// @param[in] bases Bases for elements
	/// @param[in] geom_bases Geometry bases for elements
	/// @param[in] total_local_boundary Local boundaries for elements
	/// @param[in] n_bases Number of bases (nodes)
	/// @param[in] dim Dimension of the mesh
	/// @param[in] proxy_vertices Vertices of the proxy mesh
	/// @param[out] displacement_map Output displacement map from proxy mesh to original mesh
	void build_collision_proxy_displacement_map(
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		const int n_bases,
		const int dim,
		const Eigen::MatrixXd &proxy_vertices,
		// NOTE: no need for proxy_faces
		std::vector<Eigen::Triplet<double>> &displacement_map);

	/// @brief Load a collision proxy mesh and displacement map from files.
	/// @param[in] mesh_filename Mesh filename
	/// @param[in] weights_filename Weights filename
	/// @param[in] in_node_to_node Map from input node IDs to node IDs
	/// @param[in] transformation Transformation to apply to the mesh
	/// @param[out] vertices Output vertices of the proxy mesh
	/// @param[out] codim_vertices Output codimension vertices of the proxy mesh
	/// @param[out] edges Output edges of the proxy mesh
	/// @param[out] faces Output faces of the proxy mesh
	/// @param[out] displacement_map_entries Output displacement map entries
	void load_collision_proxy(
		const std::string &mesh_filename,
		const std::string &weights_filename,
		const Eigen::VectorXi &in_node_to_node,
		const json &transformation,
		Eigen::MatrixXd &vertices,
		Eigen::VectorXi &codim_vertices,
		Eigen::MatrixXi &edges,
		Eigen::MatrixXi &faces,
		std::vector<Eigen::Triplet<double>> &displacement_map_entries);

	/// @brief Load a collision proxy mesh and displacement map from memory; the same as the file
	/// overload above from the read arrays on.
	/// @param[in] proxy Collision proxy (its collision_body_ids are not used here)
	/// @param[in] in_node_to_node Map from input node IDs to node IDs
	/// @param[in] transformation Transformation to apply to the mesh
	/// @param[out] vertices Output vertices of the proxy mesh
	/// @param[out] codim_vertices Output codimension vertices of the proxy mesh
	/// @param[out] edges Output edges of the proxy mesh
	/// @param[out] faces Output faces of the proxy mesh
	/// @param[out] displacement_map_entries Output displacement map entries
	void load_collision_proxy(
		const CollisionProxyData &proxy,
		const Eigen::VectorXi &in_node_to_node,
		const json &transformation,
		Eigen::MatrixXd &vertices,
		Eigen::VectorXi &codim_vertices,
		Eigen::MatrixXi &edges,
		Eigen::MatrixXi &faces,
		std::vector<Eigen::Triplet<double>> &displacement_map_entries);

	/// @brief Load a collision proxy mesh from a file.
	/// @param[in] mesh_filename Mesh filename
	/// @param[in] transformation Transformation to apply to the mesh
	/// @param[out] vertices Output vertices of the proxy mesh
	/// @param[out] codim_vertices Output codimension vertices of the proxy mesh
	/// @param[out] edges Output edges of the proxy mesh
	/// @param[out] faces Output faces of the proxy mesh
	void load_collision_proxy_mesh(
		const std::string &mesh_filename,
		const json &transformation,
		Eigen::MatrixXd &vertices,
		Eigen::VectorXi &codim_vertices,
		Eigen::MatrixXi &edges,
		Eigen::MatrixXi &faces);

	/// @brief Load a collision proxy displacement map from files.
	/// @param[in] weights_filename Weights filename
	/// @param[in] in_node_to_node Map from input node IDs to node IDs
	/// @param[out] displacement_map_entries Output displacement map entries
	void load_collision_proxy_displacement_map(
		const std::string &weights_filename,
		const Eigen::VectorXi &in_node_to_node,
		const size_t num_proxy_vertices,
		std::vector<Eigen::Triplet<double>> &displacement_map_entries);

	/// @brief Load per-face collision body IDs from a text file (one integer per line)
	/// and expand to per-vertex sets. Vertices shared between faces of different
	/// IDs accumulate all of those IDs.
	/// @param[in] filename Path to the collision body IDs file (n_faces lines)
	/// @param[in] faces Face connectivity matrix (n_faces x 3)
	/// @param[in] n_vertices Number of vertices
	/// @return Per-vertex sets of collision body IDs
	std::vector<std::set<int>> load_collision_proxy_collision_body_ids(
		const std::string &filename,
		const Eigen::MatrixXi &faces,
		const size_t n_vertices);

	/// @brief Expand per-face collision body IDs held in memory (one list per face, as the file
	/// holds them) to per-vertex sets; the same as the file overload above from the read lists on.
	/// @param[in] face_body_ids One list of IDs per face (n_faces lists)
	/// @param[in] faces Face connectivity matrix (n_faces x 3)
	/// @param[in] n_vertices Number of vertices
	/// @return Per-vertex sets of collision body IDs
	std::vector<std::set<int>> load_collision_proxy_collision_body_ids(
		const std::vector<std::vector<int>> &face_body_ids,
		const Eigen::MatrixXi &faces,
		const size_t n_vertices);
} // namespace polyfem::mesh
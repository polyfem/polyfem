#pragma once

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/mesh2D/PolygonMesh.hpp>
#include <polyfem/mesh/mesh3D/Navigation3D.hpp>
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace polyfem
{

	namespace mesh
	{
		class Mesh2D;
		class Mesh3D;

		///
		/// Retrieve a 3D vector with the position of a given vertex.
		///
		/// @param[in]  M      Input mesh
		/// @param[in]  v      Vertex index whose position to retrieve
		///
		/// @return     Position of the given vertex in 3D
		///
		PolygonMesh::Point mesh_vertex(const PolygonMesh &M, int v);

		// Compute facet barycenter.
		//
		// @param[in]  M      Input mesh
		// @param[in]  f      Facet whose barycenter to compute
		//
		// @return     Barycenter position in 3D
		//
		PolygonMesh::Point facet_barycenter(const PolygonMesh &M, int f);

		// Create a new mesh vertex with given coordinates.
		//
		// @param      M     Mesh to modify
		// @param[in]  p      New vertex position
		//
		// @return     Index of the newly created vertex
		//
		int mesh_create_vertex(PolygonMesh &M, const PolygonMesh::Point &p);

		///
		/// @brief      Compute the type of each facet in a surface mesh.
		///
		/// @param[in]  M              Input surface mesh (with boundary flags, see Navigation::prepare_mesh)
		/// @param[out] element_tags   Types of each facet element
		///
		void compute_element_tags(const PolygonMesh &M, std::vector<ElementType> &element_tags);

		///
		/// @brief         Orient facets of a 2D mesh so that each connected component
		///                has positive volume
		///
		/// @param[in,out] M     Surface mesh to reorient
		///
		void orient_normals_2d(PolygonMesh &M);

		///
		/// @brief         Reorder vertices of a mesh using color tags, so that vertices are ordered by
		///                increasing colors
		///
		/// @param[in,out] V     #V x d input mesh vertices
		/// @param[in,out] F     #F x k input mesh faces
		/// @param[in]     C      #V per vertex color tag
		/// @param[out]    R     max(C)+1 vector of starting indices for each colors (last value is the total number of vertices)
		///
		void reorder_mesh(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const Eigen::VectorXi &C, Eigen::VectorXi &R);

		/// Offset along area-weighted surface normals, using the direction away
		/// from an interior point where incident normals cancel.
		void offset_surface_vertices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
									 const Eigen::RowVector3d &interior_point, double distance, Eigen::MatrixXd &offset);

		///
		/// @brief      Computes the signed squared distance from a list of points to a triangle mesh. This
		///             function build a AABB tree of the input mesh, computes the distance for each query
		///             point to the closest triangle, and then determines the sign by casting a vertical
		///             ray from the query point and counting the number of intersections with the input
		///             mesh
		///
		/// @param[in]  V      #V x 3 input mesh vertices
		/// @param[in]  F      #F x 3 input mesh faces
		/// @param[in]  P      #P x 3 query points
		/// @param      D      #P x 1 computed signed distances, negative inside, positive outside
		///
		void signed_squared_distances(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
									  const Eigen::MatrixXd &P, Eigen::VectorXd &D);

		///
		/// @brief      Compute the signed volume of a surface mesh
		///
		/// @param[in]  V      #V x 3 input mesh vertices
		/// @param[in]  F      #F x 3 input mesh facets
		///
		/// @return     Signed volume of the surface
		///
		double signed_volume(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);

		///
		/// @brief      Orient a triangulated surface to have positive volume
		///
		/// @param[in]  V          #V x 3 input mesh vertices
		/// @param[in]  F          #F x 3 input mesh facets
		/// @param[in]  positive   Orient for positive volume, or negative volume
		///
		void orient_closed_surface(const Eigen::MatrixXd &V, Eigen::MatrixXi &F, bool positive = true);

		/// Pair coincident boundary edges, including nonconforming leader/follower edges.
		std::vector<std::pair<Navigation::Index, Navigation::Index>> compute_mesh_interface(
			const Mesh2D &first, const Mesh2D &second);

		/// Pair coincident boundary faces, including nonconforming leader/follower faces.
		std::vector<std::pair<Navigation3D::Index, Navigation3D::Index>> compute_mesh_interface(
			const Mesh3D &first, const Mesh3D &second);

		///
		/// @brief      Extract a set of edges that are overlap with a set given set of parent edges, using vertices positions to discriminate
		///
		/// @param[in]  IV     #IV x 3 input vertices positions
		/// @param[in]  IE     #IE x 2 input edge indices
		/// @param[in]  BV     #BV x 3 base vertices positions to test against
		/// @param[in]  BE     #BE x 2 base edge indices to test against
		/// @param[out] OE     #OE x 2 output extracted edges
		///
		void extract_parent_edges(const Eigen::MatrixXd &IV, const Eigen::MatrixXi &IE,
								  const Eigen::MatrixXd &BV, const Eigen::MatrixXi &BE, Eigen::MatrixXi &OE);

		///
		/// @brief Boundary triangles of a tet mesh, indexing the tets' vertices.
		///
		/// @param[in]  tets   #T x 4 tet indices
		/// @param[out] faces  #F x 3 boundary triangle indices
		///
		void find_triangle_surface_from_tets(const Eigen::MatrixXi &tets, Eigen::MatrixXi &faces);

		///
		/// @brief         			Extract triangular surface from a tetmesh
		///
		/// @param[in]     v,tets    tet mesh
		/// @param[out] s_v,tris 	{ surface mesh }
		///
		void extract_triangle_surface_from_tets(
			const Eigen::MatrixXd &vertices,
			const Eigen::MatrixXi &tets,
			Eigen::MatrixXd &surface_vertices,
			Eigen::MatrixXi &tris);

		/// Determine if the given #V x (2|3) vertices are planar (2D or z-range below tol).
		bool is_planar(const Eigen::MatrixXd &V, const double tol = 1e-5);

		/// Count the number of boundary elements (triangles for tetmesh and edges for triangle mesh)
		int count_faces(const int dim, const Eigen::MatrixXi &cells);

		/// @brief      Creates the edges of M (sorted by vertex pair) if it has none yet
		/// @param[in/out]  M       mesh to append edges to
		void generate_edges(PolygonMesh &M);
	} // namespace mesh
} // namespace polyfem

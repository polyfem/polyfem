#pragma once

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <geogram/mesh/mesh.h>
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace polyfem
{

	namespace mesh
	{
		///
		/// Retrieve a 3D vector with the position of a given vertex. Contrary to
		/// geogram's version, this function works with both single and double precision
		/// meshes, as well as 2D meshes too.
		///
		/// @param[in]  M      Input mesh
		/// @param[in]  v      Vertex index whose position to retrieve
		///
		/// @return     Position of the given vertex in 3D
		///
		GEO::vec3 mesh_vertex(const GEO::Mesh &M, GEO::index_t v);

		// Compute facet barycenter.
		//
		// @param[in]  M      Input mesh
		// @param[in]  f      Facet whose barycenter to compute
		//
		// @return     Barycenter position in 3D
		//
		GEO::vec3 facet_barycenter(const GEO::Mesh &M, GEO::index_t f);

		// Create a new mesh vertex with given coordinates.
		//
		// @param      M     Mesh to modify
		// @param[in]  p      New vertex position
		//
		// @return     Index of the newly created vertex
		//
		GEO::index_t mesh_create_vertex(GEO::Mesh &M, const GEO::vec3 &p);

		///
		/// @brief      Compute the type of each facet in a surface mesh.
		///
		/// @param[in]  M              Input surface mesh
		/// @param[out] element_tags   Types of each facet element
		///
		void compute_element_tags(const GEO::Mesh &M, std::vector<ElementType> &element_tags);

		///
		/// @brief         Orient facets of a 2D mesh so that each connected component
		///                has positive volume
		///
		/// @param[in,out] M     Surface mesh to reorient
		///
		void orient_normals_2d(GEO::Mesh &M);

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
		/// @brief      Converts a triangle mesh to a Geogram mesh
		///
		/// @param[in]  V      #V x 3 input mesh vertices
		/// @param[in]  F      #F x 3 input mesh surface
		/// @param[out] M      Output Geogram mesh
		///
		void to_geogram_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, GEO::Mesh &M);
		// void to_geogram_mesh_3d(const Eigen::MatrixXd &V, const Eigen::MatrixXi &C, GEO::Mesh &M);

		///
		/// @brief      Extract simplices from a Geogram mesh
		///
		/// @param[in]  M      Input Geogram mesh
		/// @param[out] V      #V x 3 output mesh vertices
		/// @param[out] F      #F x 3 output mesh faces
		/// @param[out] T      #T x 4 output mesh tets
		///
		void from_geogram_mesh(const GEO::Mesh &M, Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXi &T);

		///
		/// @brief      Converts a hex mesh to a Geogram mesh
		///
		/// @param[in]  mesh   Input mesh
		/// @param[out] M      Output Geogram mesh
		///
		void to_geogram_mesh(const Mesh3D &mesh, GEO::Mesh &M);

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

		///
		/// @brief      Extract polyhedra from a 3D volumetric mesh
		///
		/// @param[in]  mesh    Input volume mesh
		/// @param[out] polys   Extracted polyhedral surfaces
		///
		void extract_polyhedra(const Mesh3D &mesh, std::vector<std::unique_ptr<GEO::Mesh>> &polys, bool triangulated = false);

		///
		/// @brief      Tetrahedralize a star-shaped mesh, with a given point in its kernel
		///
		/// @param[in]  V        #V x 3 input mesh vertices
		/// @param[in]  F        #F x 3 input mesh triangles
		/// @param[in]  kernel   A point in the kernel
		/// @param[out] OV       #OV x 3 output mesh vertices
		/// @param[out] OF       #OF x 3 output mesh surface triangles
		/// @param[out] OT       #OT x 4 output mesh tetrahedra
		///
		void tertrahedralize_star_shaped_surface(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
												 const Eigen::RowVector3d &kernel, Eigen::MatrixXd &OV, Eigen::MatrixXi &OF, Eigen::MatrixXi &OT);

		///
		/// @brief      Samples points on a surface
		///
		/// @param[in]  V             #V x 3 input mesh vertices
		/// @param[in]  F             #F x 3 input mesh facets
		/// @param[in]  num_samples   Number of desired samples
		/// @param[out] P             num_samples x 3 sample points positions
		/// @param[out] N             num_samples x 3 of normals estimated from the original surface (optional argument)
		/// @param[in]  num_lloyd     Number of Lloyd iterations
		/// @param[in]  num_newton    Number of Newton iterations
		///
		void sample_surface(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, int num_samples,
							Eigen::MatrixXd &P, Eigen::MatrixXd *N = nullptr, int num_lloyd = 10, int num_newton = 10);

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

		///
		/// @brief      Save edge-graph into a .obj
		///
		/// @param[in]  filename   Filename to write to
		/// @param[in]  V          #V x 3 input vertices positions
		/// @param[in]  E          #E x 2 input edge indices
		///
		void save_edges(const std::string &filename, const Eigen::MatrixXd &V, const Eigen::MatrixXi &E);

		///
		/// @brief      read a mesh
		///
		/// @param[in]  mesh_path              path to mesh file
		/// @param[out] V                      #V x 3/2 output vertices positions
		/// @param[out] C                      #C cells (e.g., tri/tets/quad/hexes)
		/// @param[out] elements               #C indices for high-order nodes
		/// @param[out] w                      #C weights for rational polynomials
		///
		bool read_fem_mesh(const std::string &mesh_path, Eigen::MatrixXd &vertices, Eigen::MatrixXi &cells, std::vector<std::vector<int>> &elements, std::vector<std::vector<double>> &weights, std::vector<int> &body_ids);

		///
		/// @brief      read a surface mesh
		///
		/// @param[in]  mesh_path       path to mesh file
		/// @param[out] vertices        #V x 3/2 output vertices positions
		/// @param[out] codim_vertices  indicies in vertices for the codimensional vertices
		/// @param[out] codim_edges     indicies in vertices for the codimensional edges
		/// @param[out] faces           indicies in vertices for the surface faces
		///
		bool read_surface_mesh(const std::string &mesh_path, Eigen::MatrixXd &vertices, Eigen::VectorXi &codim_vertices, Eigen::MatrixXi &codim_edges, Eigen::MatrixXi &faces);

		/// Determine if the given mesh is planar (2D or tiny z-range).
		bool is_planar(const GEO::Mesh &M, const double tol = 1e-5);

		/// Count the number of boundary elements (triangles for tetmesh and edges for triangle mesh)
		int count_faces(const int dim, const Eigen::MatrixXi &cells);
	} // namespace mesh
} // namespace polyfem

#pragma once

#include <geogram/mesh/mesh.h>
#include <Eigen/Dense>
#include <vector>

namespace polyfem
{
	namespace mesh
	{

		// Compute a graph (V,E) where V are the edges of the input quad mesh, and E
		// connects edges from opposite sides of the input quads.
		//
		// @param[in]  Q                #Q x 4 input quads
		// @param[out] edge_index       Map (f, lv) -> edge index for edge (lv, lv+1)
		// @param[out] adj              Adjacency graph
		// @param[out] pairs_of_edges  { List of mesh edges, corresponding to the
		//                             vertices of the output graph }
		// @param[out] pairs_of_quads   List of adjacent quads
		// @param[out] quad_index      { Map (f, lv) -> index of the quad across edge
		//                             (lv, lv+1) }
		//
		void edge_adjacency_graph(const Eigen::MatrixXi &Q, Eigen::MatrixXi &edge_index,
								  std::vector<std::vector<int>> &adj,
								  std::vector<std::pair<int, int>> *pairs_of_edges = nullptr,
								  std::vector<std::pair<int, int>> *pairs_of_quads = nullptr,
								  Eigen::MatrixXi *quad_index = nullptr);

		typedef std::function<void(const Eigen::MatrixXd &, Eigen::MatrixXd &, int)> EvalParametersFunc;
		typedef std::function<std::tuple<int, int, bool>(int, int)> GetAdjacentLocalEdge;

		// Instantiate a periodic 2D pattern (triangle-mesh) on a given quad mesh
		//
		// @param[in]  IV         #IV x 3 input quad mesh vertices
		// @param[in]  IF         #IF x 4 input quad mesh facets
		// @param[in]  PV         #PV x (2|3) input pattern vertices in [0,1]^2
		// @param[in]  PF         #PF x (3|4) input pattern facets
		// @param[out] OV         #OV x 3 output mesh vertices
		// @param[out] OF         #OF x 3 output mesh facets
		// @param[in]  SF         #OV x 1 matrix of input source quad index, filled if pointer is non-zero
		// @param[in]  evalFunc  { Evaluate the uv param of the pattern into a 2d or 3d
		//                       position }
		//
		// @return     { Return true in case of success. }
		//
		bool instantiate_pattern(
			const Eigen::MatrixXd &IV, const Eigen::MatrixXi &IF,
			const Eigen::MatrixXd &PV, const Eigen::MatrixXi &PF,
			Eigen::MatrixXd &OV, Eigen::MatrixXi &OF,
			Eigen::VectorXi *SF = nullptr,
			EvalParametersFunc evalFunc = nullptr,
			GetAdjacentLocalEdge getAdjLocalEdge = nullptr);

		//
		// Refine a quad-mesh by splitting each quad into 4 quads.
		//
		// @param[in]  IV     #IV x 3 input quad mesh vertices
		// @param[in]  IF     #IF x 4 input quad mesh facets
		// @param[out] OV     #OV x 3 output mesh vertices
		// @param[out] OF     #OF x 4 output mesh facets
		//
		void refine_quad_mesh(const Eigen::MatrixXd &IV, const Eigen::MatrixXi &IF,
							  Eigen::MatrixXd &OV, Eigen::MatrixXi &OF);

		namespace Polygons
		{

			typedef std::function<void(Eigen::MatrixXd &IV, Eigen::MatrixXd &OV, std::vector<std::vector<int>> &OF)> SplitFunction;

			///
			/// Split a polygon using polar refinement. The input polygon must be
			/// star-shaped. A one-ring of quads are create on the outer ring of the
			/// polygon, while at the center a new polygonal facet is created around the
			/// barycenter of the kernel polygon. If the interpolation parameter t is
			/// equal to 0, the central polygon is collapsed into a single vertex, and
			/// facets on the ring become triangles.
			///
			/// @param[in]  IV     #IV x (2|3) of vertex positions around the polygon
			/// @param[out] OV     #OF v (2|3) output vertex positions
			/// @param[out] OF     list of output polygonal face indices
			/// @param[in]  t      Interpolation parameter to place the new vertices on the edge from the barycenter to the outer polygon vertices (0 being at the center, 1 being at the boundary).t should be >= 0.0 and < 1.0 }
			///
			void polar_split(const Eigen::MatrixXd &IV, Eigen::MatrixXd &OV, std::vector<std::vector<int>> &OF, double t = 0.5);

			/// Helper function
			inline SplitFunction polar_split_func(double t)
			{
				return [t](const Eigen::MatrixXd &IV, Eigen::MatrixXd &OV, std::vector<std::vector<int>> &OF) { polar_split(IV, OV, OF, t); };
			}

			///
			/// Split a polygon using polar refinement. A single vertex is inserted at the
			/// barycenter of the polygon kernel (the polygon needs to be star-shaped).
			/// Contrary to the `polar_split` function, this function creates quads instead
			/// of triangles. The edges of the original polygon are also assumed to be split
			/// already, so the input polygon must have an even number of vertices. Radial
			/// edges will be inserted every 2 vertices, starting from vertex #1.
			///
			/// @param[in]  IV     #IV x (2|3) of vertex positions around the polygon
			/// @param      OV     #OF v (2|3) output vertex positions
			/// @param      OF     list of output polygonal face indices
			///
			void catmul_clark_split(const Eigen::MatrixXd &IV, Eigen::MatrixXd &OV, std::vector<std::vector<int>> &OF);

			inline SplitFunction catmul_clark_split_func() { return SplitFunction(catmul_clark_split); }

			///
			/// @brief      Don't split polygons
			///
			void no_split(const Eigen::MatrixXd &IV, Eigen::MatrixXd &OV, std::vector<std::vector<int>> &OF);

			inline SplitFunction no_split_func() { return SplitFunction(no_split); }
		} // namespace Polygons

		///
		/// Refine a polygonal mesh. Quads and triangles are split into quads. If
		/// `refine_polygons` is set to `true`, then polygonal facets are also split
		/// into a layer of padding quads, and a new polygon is created around the
		/// barycenter
		///
		/// @param[in]  M_in         Surface mesh to subdivide
		/// @param[out] M_out        Refined mesh
		/// @param[in]  split_func   Functional used to split the new polygon interiors (boundary has already been split)
		///
		void refine_polygonal_mesh(const GEO::Mesh &M_in, GEO::Mesh &M_out, Polygons::SplitFunction split_func);

		///
		/// Refine a triangle mesh. Each input triangle is split into 4 new triangles
		///
		/// @param[in]  M_in    Input surface mesh
		/// @param[out] M_out   Output surface mesh
		///
		void refine_triangle_mesh(const GEO::Mesh &M_in, GEO::Mesh &M_out);

		// Kept for compatibility
		[[deprecated]] inline void refine_polygonal_mesh(const GEO::Mesh &M_in, GEO::Mesh &M_out, bool refine_polygons = false, double t = 0.5)
		{
			if (refine_polygons == false)
			{
				refine_polygonal_mesh(M_in, M_out, Polygons::no_split_func());
			}
			else
			{
				refine_polygonal_mesh(M_in, M_out, Polygons::polar_split_func(t));
			}
		}
	} // namespace mesh
} // namespace polyfem

#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/mesh/mesh2D/Navigation.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/HashUtils.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Dense>
#include <geogram/mesh/mesh.h>

#include <memory>

namespace polyfem
{
	namespace mesh
	{
		/// Type of Element, check [Poly-Spline Finite Element Method] for a complete description.
		/// **NOTE**:
		/// For the purpose of the tagging, elements (facets in 2D, cells in 3D) adjacent to a polytope
		/// are tagged as boundary, and vertices incident to a polytope are also considered as boundary.
		enum class ElementType
		{
			SIMPLEX = 0,                   /// Triangle/tet element
			REGULAR_INTERIOR_CUBE,         /// Regular quad/hex inside a 3^n patch
			SIMPLE_SINGULAR_INTERIOR_CUBE, /// Quad/hex incident to exactly 1 singular vertex (in 2D) or edge (in 3D)
			MULTI_SINGULAR_INTERIOR_CUBE,  /// Quad/Hex incident to more than 1 singular vertices (should not happen in 2D)
			REGULAR_BOUNDARY_CUBE,         /// Boundary quad/hex, where all boundary vertices/edges are incident to at most 2 quads/hexes
			SIMPLE_SINGULAR_BOUNDARY_CUBE, /// Quad incident to exactly 1 singular vertex (in 2D); hex incident to exactly 1 singular interior edge, 0 singular boundary edge, 1 boundary face (in 3D)
			MULTI_SINGULAR_BOUNDARY_CUBE,  /// Boundary hex that is not regular nor SimpleSingularBoundaryCube
			INTERFACE_CUBE,                /// Quad/hex that is at the interface with a polytope (if a cube has both external boundary and and interface with a polytope, it is marked as interface)
			INTERIOR_POLYTOPE,             /// Interior polytope
			BOUNDARY_POLYTOPE,             /// Boundary polytope
			UNDEFINED,                     /// For invalid configurations
		};

		/// Abstract mesh class to capture 2d/3d conforming and non-conforming meshes
		class Mesh
		{
		protected:
			/// Class to store the high-order edge nodes
			class EdgeNodes
			{
			public:
				int v1, v2;
				Eigen::MatrixXd nodes;
			};

			/// Class to store the high-order face nodes
			class FaceNodes
			{
			public:
				int v1, v2, v3;
				Eigen::MatrixXd nodes;
			};

			/// Class to store the high-order cells nodes
			class CellNodes
			{
			public:
				int v1, v2, v3, v4;
				Eigen::MatrixXd nodes;
			};

		public:
			///
			/// factory to build the proper mesh
			///
			/// @param[in] path mesh path
			/// @param[in] non_conforming yes or no for non conforming mesh
			/// @return pointer to the mesh
			static std::unique_ptr<Mesh> create(const std::string &path, const bool non_conforming = false);

			///
			/// factory to build the proper mesh
			///
			/// @param[in] M geo mesh
			/// @param[in] non_conforming yes or no for non conforming mesh
			/// @return pointer to the mesh
			static std::unique_ptr<Mesh> create(GEO::Mesh &M, const bool non_conforming = false);

			///
			/// factory to build the proper mesh
			///
			/// @param[in] vertices list of vertices
			/// @param[in] cells list of cells
			/// @param[in] non_conforming yes or no for non conforming mesh
			/// @return pointer to the mesh
			static std::unique_ptr<Mesh> create(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &cells, const bool non_conforming = false);

			///
			/// factory to build the proper empty mesh
			///
			/// @param[in] dim dimension of the mesh
			/// @param[in] non_conforming yes or no for non conforming mesh
			/// @return pointer to the mesh
			static std::unique_ptr<Mesh> create(const int dim, const bool non_conforming = false);

		protected:
			///
			/// @brief Construct a new Mesh object
			///
			Mesh() = default;

		public:
			/// @brief Destroy the Mesh object
			///
			virtual ~Mesh() = default;
			///
			/// @brief Construct a new Mesh object
			///
			Mesh(Mesh &&) = default;
			///
			/// @brief Copy constructor
			///
			/// @return Mesh&
			Mesh &operator=(Mesh &&) = default;
			///
			/// @brief Construct a new Mesh object
			///
			Mesh(const Mesh &) = default;
			///
			/// @brief
			///
			/// @return Mesh&
			Mesh &operator=(const Mesh &) = default;

			///
			/// @brief refine the mesh
			///
			/// @param[in] n_refinement number of refinements
			/// @param[in] t position of the refinement location (0.5 for standard refinement)
			virtual void refine(const int n_refinement, const double t) = 0;

			///
			/// @brief checks if mesh is volume
			///
			/// @return if mesh is volumetric
			virtual bool is_volume() const = 0;
			///
			/// @brief utily for dimension
			///
			/// @return int 2 or 3
			int dimension() const { return (is_volume() ? 3 : 2); }
			///
			/// @brief if the mesh is conforming
			///
			/// @return if the mesh is conforming
			virtual bool is_conforming() const = 0;
			///
			/// @brief utitlity to return the number of elements, cells or faces in 3d and 2d
			///
			/// @return number of elements
			int n_elements() const { return (is_volume() ? n_cells() : n_faces()); }
			///
			/// @brief utitlity to return the number of boundary elements, faces or edges in 3d and 2d
			///
			/// @return number of boundary elements
			int n_boundary_elements() const { return (is_volume() ? n_faces() : n_edges()); }

			///
			/// @brief number of cells
			///
			/// @return number of cells
			virtual int n_cells() const = 0;
			///
			/// @brief number of faces
			///
			/// @return number of faces
			virtual int n_faces() const = 0;
			/// @brief number of edges
			///
			/// @return number of edges
			virtual int n_edges() const = 0;
			/// @brief number of vertices
			///
			/// @return number of vertices
			virtual int n_vertices() const = 0;

			/// @brief number of vertices of a face
			///
			/// @param[in] f_id *global* face id
			/// @return  number of vertices
			virtual int n_face_vertices(const int f_id) const = 0;
			/// @brief number of vertices of a cell
			///
			/// @param[in] c_id *global* cell id (face for 2d meshes)
			/// @return  number of vertices
			virtual int n_cell_vertices(const int c_id) const = 0;
			/// @brief id of the edge vertex
			///
			/// @param[in] e_id *global* edge id
			/// @param[in] lv_id *local* vertex index
			/// @return int id of the face vertex
			virtual int edge_vertex(const int e_id, const int lv_id) const = 0;
			/// @brief id of the face vertex
			///
			/// @param[in] f_id *global* face id
			/// @param[in] lv_id *local* vertex index
			/// @return int id of the face vertex
			virtual int face_vertex(const int f_id, const int lv_id) const = 0;
			/// @brief id of the vertex of a cell
			///
			/// @param[in] f_id *global* cell id
			/// @param[in] lv_id *local* vertex id
			/// @return vertex id
			virtual int cell_vertex(const int f_id, const int lv_id) const = 0;
			/// @brief id of the vertex of a element
			///
			/// @param[in] el_id *global* element id
			/// @param[in] lv_id *local* vertex id
			/// @return vertex id
			int element_vertex(const int el_id, const int lv_id) const
			{
				return (is_volume() ? cell_vertex(el_id, lv_id) : face_vertex(el_id, lv_id));
			}

			int boundary_element_vertex(const int primitive_id, const int lv_id) const
			{
				return (is_volume() ? face_vertex(primitive_id, lv_id) : edge_vertex(primitive_id, lv_id));
			}

			/// @brief is vertex boundary
			///
			/// @param[in] vertex_global_id *global* vertex id
			/// @return is vertex boundary
			virtual bool is_boundary_vertex(const int vertex_global_id) const = 0;
			/// @brief is edge boundary
			///
			/// @param[in] edge_global_id *global* edge id
			/// @return is edge boundary
			virtual bool is_boundary_edge(const int edge_global_id) const = 0;
			/// @brief is face boundary
			///
			/// @param[in] face_global_id *global* face id
			/// @return is face boundary
			virtual bool is_boundary_face(const int face_global_id) const = 0;
			/// @brief is cell boundary
			///
			/// @param[in] element_global_id *global* cell id
			/// @return is cell boundary
			virtual bool is_boundary_element(const int element_global_id) const = 0;

			virtual bool save(const std::string &path) const = 0;

		private:
			/// @brief build a mesh from matrices
			///
			/// @param[in] V vertices
			/// @param[in] F connectivity
			/// @return if success
			virtual bool build_from_matrices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) = 0;

		public:
			/// @brief attach high order nodes
			///
			/// @param[in] V nodes
			/// @param[in] nodes list of nodes per element
			virtual void attach_higher_order_nodes(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &nodes) = 0;
			/// @brief order of each element
			///
			/// @return matrix containing order
			inline const Eigen::MatrixXi &orders() const { return orders_; }
			/// @brief check if curved mesh has rational polynomials elements
			///
			/// @return if mesh is rational
			inline bool is_rational() const { return is_rational_; }
			/// @brief Set the is rational object
			///
			/// @param[in] in_is_rational flag to enable/disable rational polynomials
			inline void set_is_rational(const bool in_is_rational) { is_rational_ = in_is_rational; }

			/// @brief normalize the mesh
			///
			virtual void normalize() = 0;

			/// @brief compute element types, see ElementType
			virtual void compute_elements_tag() = 0;
			/// @brief Update elements types
			virtual void update_elements_tag() { assert(false); }

			/// @brief edge length
			///
			/// @param[in] gid *global* edge id
			/// @return edge length
			virtual double edge_length(const int gid) const
			{
				assert(false);
				return 0;
			}
			/// @brief area of a quad face of an hex mesh
			///
			/// @param[in] gid *global* face id
			/// @return face area
			virtual double quad_area(const int gid) const
			{
				assert(false);
				return 0;
			}
			/// @brief area of a tri face of a tet mesh
			///
			/// @param[in] gid *global* face id
			/// @return tet area
			virtual double tri_area(const int gid) const
			{
				assert(false);
				return 0;
			}

			/// @brief point coordinates
			///
			/// @param[in] global_index *global* vertex index
			/// @return RowVectorNd
			virtual RowVectorNd point(const int global_index) const = 0;
			/// @brief Set the point
			///
			/// @param[in] global_index *global* vertex index
			/// @param[in] p value
			virtual void set_point(const int global_index, const RowVectorNd &p) = 0;

			/// @brief edge barycenter
			///
			/// @param[in] e *global* edge index
			/// @return edge barycenter
			virtual RowVectorNd edge_barycenter(const int e) const = 0;
			/// @brief face barycenter
			///
			/// @param[in] f *global* face index
			/// @return face barycenter
			virtual RowVectorNd face_barycenter(const int f) const = 0;
			/// @brief cell barycenter
			///
			/// @param[in] c *global* cell index
			/// @return cell barycenter
			virtual RowVectorNd cell_barycenter(const int c) const = 0;

			/// @brief all edges barycenters
			///
			/// @param[out] barycenters
			void edge_barycenters(Eigen::MatrixXd &barycenters) const;
			/// @brief all face barycenters
			///
			/// @param[out] barycenters
			void face_barycenters(Eigen::MatrixXd &barycenters) const;
			/// @brief all cells barycenters
			///
			/// @param[out] barycenters
			void cell_barycenters(Eigen::MatrixXd &barycenters) const;
			/// @brief utility for 2d/3d. In 2d it returns face_barycenters, in 3d it returns cell_barycenters
			///
			/// @param[out] barycenters the barycenters
			virtual void compute_element_barycenters(Eigen::MatrixXd &barycenters) const = 0;

			/// @brief constructs a box around every element (3d cell, 2d face)
			///
			/// @param[out] boxes axis aligned bounding boxes
			virtual void elements_boxes(std::vector<std::array<Eigen::Vector3d, 2>> &boxes) const = 0;
			/// @brief constructs barycentric coodiantes for a point p. WARNING works only for simplices
			///
			/// @param[in] p query point
			/// @param[in] el_id element id
			/// @param[out] coord matrix containing the barycentric coodinates
			virtual void barycentric_coords(const RowVectorNd &p, const int el_id, Eigen::MatrixXd &coord) const = 0;

			/// @brief computes the bbox of the mesh
			///
			/// @param[out] min min coodiante
			/// @param[out] max max coodiante
			virtual void bounding_box(RowVectorNd &min, RowVectorNd &max) const = 0;

			/// @brief checks if element is spline compatible
			///
			/// @param[in] el_id element id
			/// @return is spline compatible
			bool is_spline_compatible(const int el_id) const;
			/// @brief checks if element is cube compatible
			///
			/// @param[in] el_id element id
			/// @return is cube compatible
			bool is_cube(const int el_id) const;
			/// @brief checks if element is polygon compatible
			///
			/// @param[in] el_id element id
			/// @return is polygon compatible
			bool is_polytope(const int el_id) const;
			/// @brief checks if element is simples compatible
			///
			/// @param[in] el_id element id
			/// @return is simples compatible
			bool is_simplex(const int el_id) const;

			/// @brief Returns the elements types
			///
			/// @return vector of element types
			const std::vector<ElementType> &elements_tag() const { return elements_tag_; }
			/// @brief changes the element type
			///
			/// @param[in] el element id
			/// @param[in] type type of the element
			void set_tag(const int el, const ElementType type) { elements_tag_[el] = type; }

			/// @brief computes boundary selections based on a function
			///
			/// @param[in] marker lambda function that takes the node id, the position, and true/false if the element is on the boundary and returns an integer
			void compute_node_ids(const std::function<int(const size_t, const RowVectorNd &, bool)> &marker);

			/// @brief loads the boundary selections for a file
			///
			/// @param[in] path file's path
			virtual void load_boundary_ids(const std::string &path);
			/// @brief computes the selection based on the bbx of the mesh.
			/// Left gets 1, bottom 2, right 3, top 4, front 5, back 6
			///
			/// @param[in] eps tolerance for proximity
			virtual void compute_boundary_ids(const double eps) = 0;
			/// @brief computes boundary selections based on a function
			///
			/// @param[in] marker lambda function that takes the barycenter and returns an integer
			virtual void compute_boundary_ids(const std::function<int(const RowVectorNd &)> &marker) = 0;
			/// @brief computes boundary selections based on a function
			///
			/// @param[in] marker lambda function that takes the barycenter and true/false if the element is on the boundary and returns an integer
			virtual void compute_boundary_ids(const std::function<int(const RowVectorNd &, bool)> &marker) = 0;
			/// @brief computes boundary selections based on a function
			///
			/// @param[in] marker lambda function that takes the id, barycenter, and true/false if the element is on the boundary and returns an integer
			virtual void compute_boundary_ids(const std::function<int(const size_t, const RowVectorNd &, bool)> &marker) = 0;
			/// @brief computes boundary selections based on a function
			///
			/// @param[in] marker lambda function that takes the list of vertices and true/false if the element is on the boundary and returns an integer
			virtual void compute_boundary_ids(const std::function<int(const std::vector<int> &, bool)> &marker) = 0;
			/// @brief computes boundary selections based on a function
			///
			/// @param[in] marker lambda function that takes the id, the list of vertices, the barycenter, and true/false if the element is on the boundary and returns an integer
			virtual void compute_boundary_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &marker) = 0;

			/// @brief computes boundary selections based on a function
			///
			/// @param[in] marker lambda function that takes the id and barycenter and returns an integer
			virtual void compute_body_ids(const std::function<int(const size_t, const RowVectorNd &)> &marker) = 0;
			/// @brief Set the boundary selection from a vector
			///
			/// @param[in] boundary_ids vector one value per element
			virtual void set_boundary_ids(const std::vector<int> &boundary_ids) { boundary_ids_ = boundary_ids; }
			/// @brief Set the volume sections
			///
			/// @param[in] body_ids vector of labels, one per element
			virtual void set_body_ids(const std::vector<int> &body_ids) { body_ids_ = body_ids; }

			/// @brief Get the default boundary selection of an element (face in 3d, edge in 2d)
			///
			/// @param[in] primitive element id
			/// @return default label of element
			virtual int get_default_boundary_id(const int primitive) const
			{
				if (is_volume() ? is_boundary_face(primitive) : is_boundary_edge(primitive))
					return std::numeric_limits<int>::max(); // default for no selected boundary
				else
					return -1; // default for no boundary
			}

			/// @brief Get the boundary selection of an element (face in 3d, edge in 2d)
			///
			/// @param[in] primitive element id
			/// @return label of element
			virtual int get_boundary_id(const int primitive) const
			{
				return has_boundary_ids() ? (boundary_ids_.at(primitive)) : get_default_boundary_id(primitive);
			}

			/// @brief Get the boundary selection of a node
			///
			/// @param[in] node_id node id
			/// @return label of node
			virtual int get_node_id(const int node_id) const
			{
				if (has_node_ids())
					return node_ids_.at(node_id);
				else
					return -1; // default for no boundary
			}

			/// @brief Update the node ids to reorder them
			///
			/// @param[in] in_node_to_node mapping from input nodes to polyfem nodes
			void update_nodes(const Eigen::VectorXi &in_node_to_node);

			/// @brief Get the volume selection of an element (cell in 3d, face in 2d)
			///
			/// @param[in] primitive element id
			/// @return label of element
			virtual int get_body_id(const int primitive) const
			{
				if (has_body_ids())
					return body_ids_.at(primitive);
				else
					return 0;
			}
			/// @brief Get the volume selection of all elements (cells in 3d, faces in 2d)
			/// @return Const reference to the vector of body IDs
			virtual const std::vector<int> &get_body_ids() const
			{
				return body_ids_;
			}
			/// @brief checks if points selections are available
			///
			/// @return points selections are available
			bool has_node_ids() const { return !node_ids_.empty(); }
			/// @brief checks if surface selections are available
			///
			/// @return surface selections are available
			bool has_boundary_ids() const { return !boundary_ids_.empty(); }
			/// @brief checks if volumes selections are available
			///
			/// @return volumes selections are available
			virtual bool has_body_ids() const { return !body_ids_.empty(); }

			/// @brief Get all the edges
			///
			/// @param[out] p0 edge first vertex
			/// @param[out] p1 edge second vertex
			virtual void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1) const = 0;
			/// @brief Get all the edges according to valid_elements selection
			///
			/// @param[out] p0 edge first vertex
			/// @param[out] p1 edge second vertex
			/// @param[in] valid_elements flag to compute the edge
			virtual void get_edges(Eigen::MatrixXd &p0, Eigen::MatrixXd &p1, const std::vector<bool> &valid_elements) const = 0;
			/// @brief generate a triangular representation of every face
			///
			/// @param[out] tris triangles connectivity
			/// @param[out] pts triangles vertices
			/// @param[out] ranges connection to original faces
			virtual void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const = 0;

			/// @brief weights for rational polynomial meshes
			///
			/// @param[in] cell_index index of the cell
			/// @return list of weights
			const std::vector<double> &cell_weights(const int cell_index) const { return cell_weights_[cell_index]; }
			/// @brief Set the cell weights for rational polynomial meshes
			///
			/// @param[in] in_cell_weights vector of vector containing the weights, one per cell
			void set_cell_weights(const std::vector<std::vector<double>> &in_cell_weights) { cell_weights_ = in_cell_weights; }

			/// @brief method used to finalize the mesh. It computes the cached stuff used in navigation
			///
			virtual void prepare_mesh(){};

			/// @brief checks if the mesh has polytopes
			///
			/// @return if the mesh has polytopes
			bool has_poly() const
			{
				for (int i = 0; i < n_elements(); ++i)
				{
					if (is_polytope(i))
						return true;
				}

				return false;
			}

			/// @brief checks if the mesh is simplicial
			///
			/// @return if the mesh is simplicial
			bool is_simplicial() const
			{
				for (int i = 0; i < n_elements(); ++i)
				{
					if (!is_simplex(i))
						return false;
				}

				return true;
			}

			/// @brief check if the mesh is linear
			///
			/// @return if the mesh is linear
			inline bool is_linear() const { return orders_.size() == 0 || orders_.maxCoeff() == 1; }

			/// @brief list of *sorted* edges. Used to map to input vertices
			///
			/// @return list of *sorted* edges
			std::vector<std::pair<int, int>> edges() const;
			/// @brief list of *sorted* faces. Used to map to input vertices
			///
			/// @return list of *sorted* faces
			std::vector<std::vector<int>> faces() const;

			/// @brief map from edge (pair of v id) to the id of the edge
			///
			/// @return map
			std::unordered_map<std::pair<int, int>, size_t, polyfem::utils::HashPair> edges_to_ids() const;
			/// @brief map from face (tuple of v id) to the id of the face
			///
			/// @return map
			std::unordered_map<std::vector<int>, size_t, polyfem::utils::HashVector> faces_to_ids() const;

			/// @brief Order of the input vertices
			///
			/// @return vector of indices, one per vertex
			inline const Eigen::VectorXi &in_ordered_vertices() const { return in_ordered_vertices_; }
			/// @brief Order of the input edges
			///
			/// @return matrix of indices one per edge, pointing to the two vertices
			inline const Eigen::MatrixXi &in_ordered_edges() const { return in_ordered_edges_; }
			/// @brief Order of the input edges
			///
			/// @return matrix of indices one per faces, pointing to the face vertices
			inline const Eigen::MatrixXi &in_ordered_faces() const { return in_ordered_faces_; }

			/// @brief appends a new mesh to the end of this
			///
			/// @param[in] mesh to append
			virtual void append(const Mesh &mesh);

			/// @brief appends a new mesh to the end of this, utility that takes pointer, calls other one
			///
			/// @param[in] mesh pointer to append
			void append(const std::unique_ptr<Mesh> &mesh)
			{
				if (mesh != nullptr)
					append(*mesh);
			}

			/// @brief Apply an affine transformation \f$Ax+b\f$ to the vertex positions \f$x\f$.
			/// @param[in] A Multiplicative matrix component of transformation
			/// @param[in] b Additive translation component of transformation
			void apply_affine_transformation(const MatrixNd &A, const VectorNd &b);

		protected:
			/// @brief loads a mesh from the path
			///
			/// @param[in] path file location
			/// @return if success
			virtual bool load(const std::string &path) = 0;
			/// @brief loads a mesh from a geo mesh
			///
			/// @param[in] M geo mesh
			/// @return if success
			virtual bool load(const GEO::Mesh &M) = 0;

			/// list of element types
			std::vector<ElementType> elements_tag_;
			/// list of node labels
			std::vector<int> node_ids_;
			/// list of surface labels
			std::vector<int> boundary_ids_;
			/// list of volume labels
			std::vector<int> body_ids_;
			/// list of geometry orders, one per cell
			Eigen::MatrixXi orders_;
			/// stores if the mesh is rational
			bool is_rational_ = false;

			/// high-order nodes associates to edges
			std::vector<EdgeNodes> edge_nodes_;
			/// high-order nodes associates to faces
			std::vector<FaceNodes> face_nodes_;
			/// high-order nodes associates to cells
			std::vector<CellNodes> cell_nodes_;
			/// weights associates to cells for rational polynomail meshes
			std::vector<std::vector<double>> cell_weights_;

			/// Order of the input vertices
			Eigen::VectorXi in_ordered_vertices_;
			/// Order of the input edges
			Eigen::MatrixXi in_ordered_edges_;
			/// Order of the input faces, TODO: change to std::vector of Eigen::Vector
			Eigen::MatrixXi in_ordered_faces_;
		};
	} // namespace mesh
} // namespace polyfem

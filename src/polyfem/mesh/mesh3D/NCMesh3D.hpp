#pragma once

#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <set>
#include <tuple>

namespace polyfem
{
	namespace mesh
	{
		class NCMesh3D : public Mesh3D
		{
		public:
			struct follower_edge
			{
				int id;
				double p1, p2;

				follower_edge(int id_, double p1_, double p2_)
				{
					id = id_;
					p1 = p1_;
					p2 = p2_;
				}
			};

			struct follower_face
			{
				int id;
				Eigen::Vector2d p1, p2, p3;

				follower_face(int id_, Eigen::Vector2d p1_, Eigen::Vector2d p2_, Eigen::Vector2d p3_)
				{
					id = id_;
					p1 = p1_;
					p2 = p2_;
					p3 = p3_;
				}
			};

			struct ncVert
			{
				ncVert(const Eigen::VectorXd pos_) : pos(pos_){};
				~ncVert(){};

				int n_elem() const
				{
					return elem_list.size();
				}

				void add_element(const int e)
				{
					elem_list.insert(e);
				};

				void remove_element(const int e)
				{
					int num = elem_list.erase(e);
					assert(num == 1);
				}

				std::set<int> elem_list; // elements that contain this edge/face

				Eigen::VectorXd pos;
				bool isboundary = false;

				int edge = -1; // only used if the vertex is on the interior of an edge
				int face = -1; // only used if the vertex is on the interior of an face

				double weight = -1.; // only 2d, the local position of this vertex on the edge
			};

			struct ncBoundary
			{
				ncBoundary(const Eigen::VectorXi vert) : vertices(vert)
				{
					weights.setConstant(-1);
				};
				~ncBoundary(){};

				int n_elem() const
				{
					return elem_list.size();
				}

				void add_element(const int e)
				{
					elem_list.insert(e);
				};

				void remove_element(const int e)
				{
					int num = elem_list.erase(e);
					assert(num == 1);
				}

				int get_element() const
				{
					assert(n_elem() > 0);
					auto it = elem_list.cbegin();
					return *it;
				}

				int find_opposite_element(int e) const
				{
					assert(n_elem() == 2);
					bool exist = false;
					int oppo = -1;
					for (int elem : elem_list)
						if (elem == e)
							exist = true;
						else
							oppo = elem;

					assert(oppo >= 0);
					assert(exist);
					return oppo;
				}

				std::set<int> elem_list; // elements that contain this edge/face
				Eigen::VectorXi vertices;

				bool isboundary = false; // valid only after calling mark_boundary()
				int boundary_id = 0;

				int leader = -1;            // if this edge/face lies on a larger edge/face
				std::vector<int> followers; // followers of this edge/face

				int leader_face = -1; // if this edge lies in the interior of a face

				std::vector<int> global_ids; // only used for building basis

				// the following only used if it's an edge
				Eigen::Vector2d weights; // position of this edge on its leader edge
			};

			struct ncElem
			{
				ncElem(const int dim_, const Eigen::VectorXi vertices_, const int level_, const int parent_) : dim(dim_), level(level_), parent(parent_), geom_vertices(vertices_)
				{
					vertices = geom_vertices;

					assert(geom_vertices.size() == dim + 1);
					edges.setConstant(3 * (dim - 1), 1, -1);
					faces.setConstant(4 * (dim - 2), 1, -1);
					children.setConstant(std::round(pow(2, dim)), 1, -1);
				};
				~ncElem(){};

				bool is_valid() const
				{
					return (!is_ghost) && (!is_refined);
				};

				bool is_not_valid() const
				{
					return is_ghost || is_refined;
				};

				int dim;
				int level; // level of refinement
				int parent;
				Eigen::VectorXi geom_vertices;

				Eigen::VectorXi vertices; // geom_vertices with a different order that used in polyfem

				Eigen::VectorXi edges;
				Eigen::VectorXi faces;
				Eigen::VectorXi children;

				int body_id;

				bool is_refined = false;
				bool is_ghost = false;
			};

			NCMesh3D() = default;
			virtual ~NCMesh3D() = default;
			NCMesh3D(NCMesh3D &&) = default;
			NCMesh3D &operator=(NCMesh3D &&) = default;
			NCMesh3D(const NCMesh3D &) = default;
			NCMesh3D &operator=(const NCMesh3D &) = default;

			void refine(const int n_refinement, const double t) override;

			bool is_conforming() const override { return false; }

			int n_cells() const override { return n_elements; }
			int n_faces() const override
			{
				int n = 0;
				for (const auto &face : faces)
					if (face.n_elem())
						n++;
				return n;
			}
			int n_edges() const override
			{
				int n = 0;
				for (const auto &edge : edges)
					if (edge.n_elem())
						n++;
				return n;
			}
			int n_vertices() const override
			{
				int n = 0;
				for (const auto &vert : vertices)
					if (vert.n_elem())
						n++;
				return n;
			}

			int n_face_vertices(const int f_id) const override { return 3; }
			int n_face_cells(const int f_id) const { return faces[valid_to_all_face(f_id)].n_elem(); }
			int n_edge_cells(const int e_id) const { return edges[valid_to_all_edge(e_id)].n_elem(); }
			int n_cell_vertices(const int c_id) const override { return 4; }
			int n_cell_edges(const int c_id) const override { return 6; }
			int n_cell_faces(const int c_id) const override { return 4; }
			int cell_vertex(const int f_id, const int lv_id) const override { return all_to_valid_vertex(elements[valid_to_all_elem(f_id)].vertices(lv_id)); }
			int cell_face(const int c_id, const int lf_id) const override { return all_to_valid_face(elements[valid_to_all_elem(c_id)].faces(lf_id)); }
			int cell_edge(const int c_id, const int le_id) const override { return all_to_valid_edge(elements[valid_to_all_elem(c_id)].edges(le_id)); }
			int face_vertex(const int f_id, const int lv_id) const override { return all_to_valid_vertex(faces[valid_to_all_face(f_id)].vertices(lv_id)); }
			int face_edge(const int f_id, const int le_id) const;
			int edge_vertex(const int e_id, const int lv_id) const override { return all_to_valid_vertex(edges[valid_to_all_edge(e_id)].vertices(lv_id)); }

			inline int cell_ref_level(const int c_id) const { return elements[valid_to_all_elem(c_id)].level; }

			bool is_boundary_vertex(const int vertex_global_id) const override { return vertices[valid_to_all_vertex(vertex_global_id)].isboundary; }
			bool is_boundary_edge(const int edge_global_id) const override { return edges[valid_to_all_edge(edge_global_id)].isboundary; }
			bool is_boundary_face(const int face_global_id) const override { return faces[valid_to_all_face(face_global_id)].isboundary; }
			bool is_boundary_element(const int element_global_id) const override;

			bool save(const std::string &path) const override 
			{ 
				// TODO 
				return false;
			}

			bool build_from_matrices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) override;

			void attach_higher_order_nodes(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &nodes) override;

			void normalize() override;

			void compute_elements_tag() override;
			void update_elements_tag() override;

			double edge_length(const int gid) const override;

			RowVectorNd point(const int global_index) const override;
			void set_point(const int global_index, const RowVectorNd &p) override;
			RowVectorNd edge_barycenter(const int e) const override;
			RowVectorNd face_barycenter(const int f) const override;
			RowVectorNd cell_barycenter(const int c) const override;

			void bounding_box(RowVectorNd &min, RowVectorNd &max) const override;

			void compute_boundary_ids(const double eps) override;
			void compute_boundary_ids(const std::function<int(const RowVectorNd &)> &marker) override;
			void compute_boundary_ids(const std::function<int(const RowVectorNd &, bool)> &marker) override;
			void compute_boundary_ids(const std::function<int(const size_t, const RowVectorNd &, bool)> &marker) override;
			void compute_boundary_ids(const std::function<int(const std::vector<int> &, bool)> &marker) override;
			void compute_boundary_ids(const std::function<int(const size_t, const std::vector<int> &, const RowVectorNd &, bool)> &marker) override;

			void compute_body_ids(const std::function<int(const size_t, const RowVectorNd &)> &marker) override;
			void set_boundary_ids(const std::vector<int> &boundary_ids) override;
			void set_body_ids(const std::vector<int> &body_ids) override;

			int get_boundary_id(const int primitive) const override { return faces[valid_to_all_face(primitive)].boundary_id; };
			int get_body_id(const int primitive) const override { return elements[valid_to_all_elem(primitive)].body_id; };

			void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const override;

			RowVectorNd kernel(const int cell_id) const override;
			// navigation wrapper
			Navigation3D::Index get_index_from_element(int hi, int lf, int lv) const override;
			Navigation3D::Index get_index_from_element(int hi) const override;

			Navigation3D::Index get_index_from_element_edge(int hi, int v0, int v1) const override;
			Navigation3D::Index get_index_from_element_face(int hi, int v0, int v1, int v2) const override;

			std::vector<uint32_t> vertex_neighs(const int v_gid) const override;
			std::vector<uint32_t> edge_neighs(const int e_gid) const override;

			int leader_edge_of_vertex(const int v_id) const
			{
				assert(adj_prepared);
				return (vertices[valid_to_all_vertex(v_id)].edge < 0) ? -1 : all_to_valid_edge(vertices[valid_to_all_vertex(v_id)].edge);
			}
			int leader_edge_of_edge(const int e_id) const
			{
				assert(adj_prepared);
				return (edges[valid_to_all_edge(e_id)].leader < 0) ? -1 : all_to_valid_edge(edges[valid_to_all_edge(e_id)].leader);
			}

			int leader_face_of_vertex(const int v_id) const
			{
				assert(adj_prepared);
				int id_full = vertices[valid_to_all_vertex(v_id)].face;
				return (id_full < 0) ? -1 : all_to_valid_face(id_full);
			}
			int leader_face_of_edge(const int e_id) const
			{
				assert(adj_prepared);
				int id_full = edges[valid_to_all_edge(e_id)].leader_face;
				return (id_full < 0) ? -1 : all_to_valid_face(id_full);
			}
			int leader_face_of_face(const int f_id) const
			{
				assert(adj_prepared);
				int id_full = faces[valid_to_all_face(f_id)].leader;
				return (id_full < 0) ? -1 : all_to_valid_face(id_full);
			}

			// number of follower edges of a leader edge
			int n_follower_edges(const int e_id) const
			{
				assert(adj_prepared);
				return edges[valid_to_all_edge(e_id)].followers.size();
			}
			int n_follower_faces(const int e_id) const
			{
				assert(adj_prepared);
				return faces[valid_to_all_face(e_id)].followers.size();
			}

			// Navigation in a surface mesh
			Navigation3D::Index switch_vertex(Navigation3D::Index idx) const override;
			Navigation3D::Index switch_edge(Navigation3D::Index idx) const override;
			Navigation3D::Index switch_face(Navigation3D::Index idx) const override;
			Navigation3D::Index switch_element(Navigation3D::Index idx) const override;

			// Iterate in a mesh
			Navigation3D::Index next_around_edge(Navigation3D::Index idx) const override;
			Navigation3D::Index next_around_face(Navigation3D::Index idx) const override;

			void get_vertex_elements_neighs(const int v_id, std::vector<int> &ids) const override;
			void get_edge_elements_neighs(const int e_id, std::vector<int> &ids) const override;
			void get_face_elements_neighs(const int f_id, std::vector<int> &ids) const;

			void refine_element(int id_full);
			void refine_elements(const std::vector<int> &ids);

			void coarsen_element(int id_full);

			void mark_boundary();

			void prepare_mesh() override
			{
				build_edge_follower_chain();
				build_face_follower_chain();
				build_element_vertex_adjacency();
				build_index_mapping();
				compute_elements_tag();
				mark_boundary();
				adj_prepared = true;
			}

			void build_index_mapping();

			std::array<int, 4> get_ordered_vertices_from_tet(const int element_index) const override;

			void append(const Mesh &mesh) override;

		private:
			struct ArrayHasher2D
			{
				long operator()(const Eigen::Vector2i &a) const
				{
					return (long)((long)984120265 * a[0] + (long)125965121 * a[1]);
				}
			};

			struct ArrayHasher3D
			{
				long operator()(const Eigen::Vector3i &a) const
				{
					return (long)((long)984120265 * a[0] + (long)125965121 * a[1] + (long)495698413 * a[2]);
				}
			};

		protected:
			bool load(const std::string &path) override;
			bool load(const GEO::Mesh &M) override;

			// index map from vertices to valid ones, and its inverse
			inline int all_to_valid_vertex(const int id) const
			{
				assert(index_prepared);
				return all_to_valid_vertexMap[id];
			};
			inline int valid_to_all_vertex(const int id) const
			{
				assert(index_prepared);
				return valid_to_all_vertexMap[id];
			};

			// index map from edges to valid ones, and its inverse
			inline int all_to_valid_edge(const int id) const
			{
				assert(index_prepared);
				return all_to_valid_edgeMap[id];
			};
			inline int valid_to_all_edge(const int id) const
			{
				assert(index_prepared);
				return valid_to_all_edgeMap[id];
			};

			// index map from faces to valid ones, and its inverse
			inline int all_to_valid_face(const int id) const
			{
				assert(index_prepared);
				return all_to_valid_faceMap[id];
			};
			inline int valid_to_all_face(const int id) const
			{
				assert(index_prepared);
				return valid_to_all_faceMap[id];
			};

			// index map from elements to valid ones, and its inverse
			inline int all_to_valid_elem(const int id) const
			{
				assert(index_prepared);
				return all_to_valid_elemMap[id];
			};
			inline int valid_to_all_elem(const int id) const
			{
				assert(index_prepared);
				return valid_to_all_elemMap[id];
			};

			// find the mid-point of edge v[0]v[1], return -1 if not exists
			int find_vertex(Eigen::Vector2i v) const;
			int find_vertex(const int v1, const int v2) const { return find_vertex(Eigen::Vector2i(v1, v2)); };

			// find the mid-point of edge v[0]v[1], create one if not exists
			int get_vertex(Eigen::Vector2i v);

			// find the edge v[0]v[1], return -1 if not exists
			int find_edge(Eigen::Vector2i v) const;
			int find_edge(const int v1, const int v2) const { return find_edge(Eigen::Vector2i(v1, v2)); };

			// find the edge v[0]v[1], create one if not exists
			int get_edge(Eigen::Vector2i v);
			int get_edge(const int v1, const int v2) { return get_edge(Eigen::Vector2i(v1, v2)); };

			// find the face, return -1 if not exists
			int find_face(Eigen::Vector3i v) const;
			int find_face(const int v1, const int v2, const int v3) const { return find_face(Eigen::Vector3i(v1, v2, v3)); };
			// find the face, create one if not exists
			int get_face(Eigen::Vector3i v);
			int get_face(const int v1, const int v2, const int v3) { return get_face(Eigen::Vector3i(v1, v2, v3)); };

			void traverse_edge(Eigen::Vector2i v, double p1, double p2, int depth, std::vector<follower_edge> &list) const;
			void build_edge_follower_chain();

			void traverse_face(int v1, int v2, int v3, Eigen::Vector2d p1, Eigen::Vector2d p2, Eigen::Vector2d p3, int depth, std::vector<follower_face> &face_list, std::vector<int> &edge_list) const;
			void build_face_follower_chain();

			void build_element_vertex_adjacency();

			int add_element(Eigen::Vector4i v, int parent = -1);

			int n_elements = 0;

			bool index_prepared = false;
			bool adj_prepared = false;

			std::vector<ncElem> elements;
			std::vector<ncVert> vertices;
			std::vector<ncBoundary> edges;
			std::vector<ncBoundary> faces;

			std::unordered_map<Eigen::Vector2i, int, ArrayHasher2D> midpointMap;
			std::unordered_map<Eigen::Vector2i, int, ArrayHasher2D> edgeMap;
			std::unordered_map<Eigen::Vector3i, int, ArrayHasher3D> faceMap;

			std::vector<int> all_to_valid_elemMap, valid_to_all_elemMap;
			std::vector<int> all_to_valid_vertexMap, valid_to_all_vertexMap;
			std::vector<int> all_to_valid_edgeMap, valid_to_all_edgeMap;
			std::vector<int> all_to_valid_faceMap, valid_to_all_faceMap;

			std::vector<int> refineHistory;

			// elementAdj(i, j) = 1 iff element i touches element j
			Eigen::SparseMatrix<bool, Eigen::RowMajor> elementAdj;
		};
	} // namespace mesh
} // namespace polyfem

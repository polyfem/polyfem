#pragma once

#include <polyfem/Mesh2D.hpp>

namespace polyfem
{

struct ArrayHasher2D 
{
    long operator()(const Eigen::Vector2i& a) const {
        return (long)((long)984120265 * a[0] + (long)125965121 * a[1]);
    }
};

struct ArrayHasher3D 
{
    long operator()(const Eigen::Vector3i& a) const {
        return (long)((long)984120265 * a[0] + (long)125965121 * a[1] + (long)495698413 * a[2]);
    }
};

class NCMesh2D: public Mesh2D
{
public:
    struct slave_edge {
        int id;
        double p1, p2;

        slave_edge(int id_, double p1_, double p2_)
        { id = id_; p1 = p1_; p2 = p2_; }
    };

    struct slave_face {
        int id;
        Eigen::Vector2d p1, p2, p3;

        slave_face(int id_, Eigen::Vector2d p1_, Eigen::Vector2d p2_, Eigen::Vector2d p3_)
        { id = id_; p1 = p1_; p2 = p2_; p3 = p3_; }
    };

    struct ncVert
    {
        ncVert(const Eigen::VectorXd pos_) : pos(pos_) {};
        ~ncVert() {};

        Eigen::VectorXd pos;
        bool isboundary = false;
        
        int edge = -1;          // only used if the vertex is on the interior of an edge
        int face = -1;          // only 3d, only used if the vertex is on the interior of an face
        
        double weight = -1.;    // only 2d, the local position of this vertex on the edge

        int n_elem = 0;         // number of valid elements that share this vertex
        bool flag = false;      // vertex flag for determining singularity
    };

    struct ncBoundary 
    {
        ncBoundary(const Eigen::VectorXi vert) : vertices(vert)
        { 
            weights.setConstant(-1);
        };
        ~ncBoundary() {};

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

        std::set<int>   elem_list; // elements that contain this edge/face
        Eigen::VectorXi vertices;

        bool  isboundary = false; // valid only after calling markBoundary()

        int               master = -1; // if this edge/face lies on a larger edge/face
        std::vector<int>  slaves;      // slaves of this edge/face

        int               master_face = -1; // if this edge lies in the interior of a face

        std::vector<int>  global_ids;  // only used for building basis

        bool flag = false;           // flag for determining singularity
        bool boundary_flag = 0;      // boundary id of this edge/face, for inheriting boundary condtion

        // the following only used if it's an edge
        Eigen::Vector2d   weights;     // position of this edge on its master edge
    };

    struct ncElem
    {
        ncElem(const int dim_, const Eigen::VectorXi vertices_, const int level_, const int parent_) : dim(dim_), level(level_), parent(parent_), geom_vertices(vertices_)
        {
            vertices = geom_vertices;
            
            assert(geom_vertices.size() == dim + 1);
            edges.setConstant(3*(dim - 1), 1, -1);
            faces.setConstant(4*(dim-2), 1, -1);
            children.setConstant(std::round(pow(2, dim)), 1, -1);
        };
        ~ncElem() {};

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

        bool   is_refined = false;
        bool   is_ghost = false;
    };

    NCMesh2D() = default;
    virtual ~NCMesh2D() = default;
    POLYFEM_DEFAULT_MOVE_COPY(NCMesh2D)

    bool is_conforming() const override { return false; }

    int n_faces() const override { return n_elements; }
    int n_edges() const override
    {
        int n = 0;
        for (const auto& edge : edges)
            if (edge.n_elem())
                n++;
        return n;
    }
    int n_vertices() const override
    {
        int n_verts = 0;
        for (const auto& vert : vertices)
            if (vert.n_elem)
                n_verts++;

        return n_verts;
    }

    inline int n_face_vertices(const int f_id) const { return 3; }

    inline int face_ref_level(const int f_id) const { return elements[valid2AllElem(f_id)].level; }

    int face_vertex(const int f_id, const int lv_id) const override { return all2ValidVertex(elements[valid2AllElem(f_id)].vertices(lv_id)); }
    int edge_vertex(const int e_id, const int lv_id) const override { return all2ValidVertex(edges[valid2AllEdge(e_id)].vertices(lv_id)); }
    int cell_vertex(const int f_id, const int lv_id) const override { return all2ValidVertex(elements[valid2AllElem(f_id)].vertices(lv_id)); }

    int face_edge(const int f_id, const int le_id) const { return all2ValidEdge(elements[valid2AllElem(f_id)].edges(le_id)); }
    
    int master_edge_of_vertex(const int v_id) const { return (vertices[valid2AllVertex(v_id)].edge < 0) ? -1 : all2ValidEdge(vertices[valid2AllVertex(v_id)].edge); }
    int master_edge_of_edge(const int e_id) const { return (edges[valid2AllEdge(e_id)].master < 0) ? -1 : all2ValidEdge(edges[valid2AllEdge(e_id)].master); }
    
    // number of slave edges of a master edge
    int n_slave_edges(const int e_id) const { return edges[valid2AllEdge(e_id)].slaves.size(); }
    // number of elements have this edge
    int n_face_neighbors(const int e_id) const { return edges[valid2AllEdge(e_id)].n_elem(); }
    // return the only element that has this edge
    int face_neighbor(const int e_id) const { return all2ValidElem(edges[valid2AllEdge(e_id)].get_element()); }

    bool is_boundary_vertex(const int vertex_global_id) const override
    {
        return vertices[vertex_global_id].isboundary;
    }

    bool is_boundary_edge(const int edge_global_id) const override
    {
        return edges[edge_global_id].isboundary;
    }

    bool is_boundary_element(const int element_global_id) const override
    {
        const auto& elem = elements[element_global_id];
        for (int le = 0; le < elem.edges.size(); le++)
            if (is_boundary_edge(elem.edges(le)))
                return true;
        
        return false;
    }

	void refine(const int n_refiniment, const double t, std::vector<int> &parent_nodes) override
    {
        std::vector<bool> refine_mask(elements.size(), false);
        for (int i = 0; i < elements.size(); i++)
            if (elements[i].is_valid())
                refine_mask[i] = true;

        for (int i = 0; i < refine_mask.size(); i++)
            if (refine_mask[i])
                refineElement(i);

        refine(n_refiniment - 1, t, parent_nodes);
    };

    bool build_from_matrices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) override;
    bool save(const std::string &path) const override;

    void attach_higher_order_nodes(const Eigen::MatrixXd &V, const std::vector<std::vector<int>> &nodes) override
    {
        for (int f = 0; f < n_faces(); ++f)
            if (nodes[f].size() != 3)
                throw std::runtime_error("NCMesh doesn't support high order mesh!");
    }
    RowVectorNd edge_node(const Navigation::Index &index, const int n_new_nodes, const int i) const override
    {
        const auto v1 = point(index.vertex);
        const auto v2 = point(switch_vertex(index).vertex);

        const double t = i / (n_new_nodes + 1.0);

        return (1 - t) * v1 + t * v2;
    }
    RowVectorNd face_node(const Navigation::Index &index, const int n_new_nodes, const int i, const int j) const override
    {
        const auto v1 = point(index.vertex);
        const auto v2 = point(switch_vertex(index).vertex);
        const auto v3 = point(switch_vertex(switch_edge(index)).vertex);

        const double b2 = i / (n_new_nodes + 2.0);
        const double b3 = j / (n_new_nodes + 2.0);
        const double b1 = 1 - b3 - b2;
        assert(b3 < 1);
        assert(b3 > 0);

        return b1 * v1 + b2 * v2 + b3 * v3;
    }

    void normalize() override;

    double edge_length(const int gid) const override;

    void compute_elements_tag() override;
    void update_elements_tag() override;

    void set_point(const int global_index, const RowVectorNd &p) override;

    RowVectorNd point(const int global_index) const override { return vertices[valid2AllVertex(global_index)].pos.transpose(); }
    RowVectorNd edge_barycenter(const int index) const override;

    void bounding_box(RowVectorNd &min, RowVectorNd &max) const override;

    // Navigation wrapper
    Navigation::Index get_index_from_face(int f, int lv = 0) const override;

    // Navigation in a surface mesh
    Navigation::Index switch_vertex(Navigation::Index idx) const override;
    Navigation::Index switch_edge(Navigation::Index idx) const override;
    Navigation::Index switch_face(Navigation::Index idx) const override;

    void triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const override;

    // edges are created if not exist
    // return the id of this new element
    int addElement(Eigen::Vector3i v, int parent = -1);
    
    // refine one element
    void refineElement(int id);

    // coarsen one element
    void coarsenElement(int id);

    // find the local index of edges[l] in the elements[e]
    int globalEdge2LocalEdge(const int e, const int l) const;

    // mark the true boundary vertices
    void markBoundary();

    // map the weight on edge to the barycentric coordinate in element
    static Eigen::Vector2d edgeWeight2ElemWeight(const int l, const double w);
    // map the barycentric coordinate in element to the weight on edge
    static double elemWeight2EdgeWeight(const int l, const Eigen::Vector2d& pos);

    // list all slave edges of a potential master edge, returns nothing if it's a slave or conforming edge
    void traverseEdge(Eigen::Vector2i v, double p1, double p2, int depth, std::vector<slave_edge>& list) const;

    // call traverseEdge() for every interface, and store everything needed
    void buildEdgeSlaveChain();

    // assign ncElement2D.master_edges and ncVertex2D.weight
    void buildElementVertexAdjacency();

    // call necessary functions before building bases
    void prepareMesh()
    {
        buildEdgeSlaveChain();
        buildElementVertexAdjacency();
        buildIndexMapping();
    }

    // find the mid-point of edge v[0]v[1], return -1 if not exists
    int findVertex(Eigen::Vector2i v) const
    {
        std::sort(v.data(), v.data()+v.size());
        auto search = midpointMap.find(v);
        if (search != midpointMap.end())
            return search->second;
        else
            return -1;
    };
    int findVertex(const int v1, const int v2) const { return findVertex(Eigen::Vector2i(v1, v2)); };
    // find the mid-point of edge v[0]v[1], create one if not exists
    int getVertex(Eigen::Vector2i v)
    {
        std::sort(v.data(), v.data()+v.size());
        int id = findVertex(v);
        if (id < 0) {
            Eigen::VectorXd v_mid = (vertices[v[0]].pos + vertices[v[1]].pos) / 2.;
            id = vertices.size();
            vertices.emplace_back(v_mid);
            midpointMap.emplace(v, id);
        }
        return id;
    };

    // find the edge v[0]v[1], return -1 if not exists
    int findEdge(Eigen::Vector2i v) const
    {
        std::sort(v.data(), v.data()+v.size());
        auto search = edgeMap.find(v);
        if (search != edgeMap.end())
            return search->second;
        else 
            return -1;
    };
    int findEdge(const int v1, const int v2) const { return findEdge(Eigen::Vector2i(v1, v2)); };

    // find the edge v[0]v[1], create one if not exists
    int getEdge(Eigen::Vector2i v)
    {
        std::sort(v.data(), v.data()+v.size());
        int id = findEdge(v);
        if (id < 0) {
            edges.emplace_back(v);
            id = edges.size() - 1;
            edgeMap.emplace(v, id);
        }
        return id;
    };
    int getEdge(const int v1, const int v2) { return getEdge(Eigen::Vector2i(v1, v2)); };

    // index map from vertices to valid ones, and its inverse
    inline int all2ValidVertex(const int id) const { return all2ValidVertexMap[id]; };
    inline int valid2AllVertex(const int id) const { return valid2AllVertexMap[id]; };

    // index map from edges to valid ones, and its inverse
    inline int all2ValidEdge(const int id) const { return all2ValidEdgeMap[id]; };
    inline int valid2AllEdge(const int id) const { return valid2AllEdgeMap[id]; };

    // index map from elements to valid ones, and its inverse
    inline int all2ValidElem(const int id) const { return all2ValidMap[id]; };
    inline int valid2AllElem(const int id) const { return valid2AllMap[id]; };

    void buildIndexMapping();
    
protected:
    bool load(const std::string &path) override;
    bool load(const GEO::Mesh &mesh) override;

    int n_elements = 0;

    std::vector<ncElem> elements;
    std::vector<ncVert> vertices;
    std::vector<ncBoundary> edges;

    std::unordered_map< Eigen::Vector2i, int, ArrayHasher2D >  midpointMap;
    std::unordered_map< Eigen::Vector2i, int, ArrayHasher2D >  edgeMap;

    std::vector<int> all2ValidMap, valid2AllMap;
    std::vector<int> all2ValidVertexMap, valid2AllVertexMap;
    std::vector<int> all2ValidEdgeMap, valid2AllEdgeMap;

    std::vector<int> refineHistory;

    // elementAdj(i, j) = 1 iff element i touches element j
    Eigen::SparseMatrix<bool, Eigen::RowMajor> elementAdj;
};

} // namespace polyfem

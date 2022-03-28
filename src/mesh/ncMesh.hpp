#ifndef NCMESH_HPP
#define NCMESH_HPP

#include <unordered_map>
#include <array>
#include <vector>
#include <string>
#include <assert.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <polyfem/Common.hpp>
#include <set>

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

class ncMesh
{
public:
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

public:
    ncMesh() { n_elements = 0; };
    virtual ~ncMesh() = default;

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

    virtual int dim() const = 0;

    virtual void init(const Eigen::MatrixXd& v, const Eigen::MatrixXi& f) = 0;
    virtual void init(const std::string& path) = 0;
    void initFromHistory(const json& input);
    
    // normalize the mesh
    void normalize();
    
    // refine one element
    virtual void refineElement(int id) = 0;
    void refineElement(int id, int n_ref)
    {
        if (n_ref <= 0)
            return;
        refineElement(id);
        for (int i = 0; i < elements[id].children.size(); i++)
            refineElement(elements[id].children(i), n_ref - 1);
    }
    // uniform refinement
    void refine()
    {
        std::vector<bool> refine_mask(elements.size(), false);
        for (int i = 0; i < elements.size(); i++)
            if (elements[i].is_valid())
                refine_mask[i] = true;

        for (int i = 0; i < refine_mask.size(); i++)
            if (refine_mask[i])
                refineElement(i);
    }

    // coarsen one element
    virtual void coarsenElement(int id) = 0;

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

    // find the face, return -1 if not exists
    virtual int findFace(Eigen::Vector3i v) const
    {
        assert(false);
        return 0;
    };
    virtual int findFace(const int v1, const int v2, const int v3) const
    {
        assert(false);
        return 0;
    };
    // find the face, create one if not exists
    virtual int getFace(Eigen::Vector3i v)
    {
        assert(false);
        return 0;
    };
    virtual int getFace(const int v1, const int v2, const int v3)
    {
        assert(false);
        return 0;
    };

    // create the tri-mesh with gaps
    void compress(Eigen::MatrixXd& v, Eigen::MatrixXi& f) const;

    // build a partial mesh including a subset of elements
    void buildPartialMesh(const std::vector<int>& elem_list, Eigen::MatrixXd& v, Eigen::MatrixXi& f) const;

    // mark the true boundary vertices
    virtual void markBoundary() = 0;

    // index map from elements to valid ones, and its inverse
    int all2Valid(const int id) const { return all2ValidMap[id]; };
    int valid2All(const int id) const { return valid2AllMap[id]; };

    // primitive metric
    double edgeLength(int e) const
    {
        assert(e >= 0 && e < edges.size());
        return (vertices[edges[e].vertices(0)].pos - vertices[edges[e].vertices(1)].pos).norm();
    };

    // index map from vertices to valid ones, and its inverse
    int all2ValidVertex(const int id) const { return all2ValidVertexMap[id]; };
    int valid2AllVertex(const int id) const { return valid2AllVertexMap[id]; };

    // count the number of valid elements of different levels
    void getLevels(Eigen::VectorXi& levels, const int max_level) const
    {
        levels.setConstant(max_level+1, 1, 0);

        for (const auto& e : elements)
            if (e.is_valid())
                levels[e.level]++;
        assert(levels.sum() == n_elements);
    };

    // build the sparse matrix Adj
    virtual void buildElementAdj(int ring = 1) = 0;
    // find all valid elements that touches elements[valid2All(e)], including itself
    void findAdjacentElements(int e, std::vector<int>& list) const
    {
        list.clear();
        for (Eigen::SparseMatrix<bool, Eigen::RowMajor>::InnerIterator it(elementAdj, e); it; ++it) {
            if (it.value())
                list.push_back(valid2All(it.col()));
        }
    };
    // put all valid children of element e into list
    void getValidChildren(int e, std::vector<int>& list) const
    {
        const auto& elem = elements[e];
        if (elem.is_refined)
            for (int i = 0; i < elem.children.size(); i++)
                getValidChildren(elem.children(i), list);
        else
            list.push_back(e);
    };

    // list all slave edges of a potential master edge, returns nothing if it's a slave or conforming edge
    void traverseEdge(Eigen::Vector2i v, double p1, double p2, int depth, std::vector<slave_edge>& list) const;

    // call traverseEdge() for every interface, and store everything needed
    virtual void buildEdgeSlaveChain() = 0;

    // 3d only, call traverseFace() for every valid face, and store everything needed
    virtual void buildFaceSlaveChain() { };

    // assign ncElement.master_edges and ncVertex.weight
    virtual void buildElementVertexAdjacency() = 0;

    // create all2ValidMap and valid2AllMap for future use
    void buildIndexMapping();

    // call necessary functions before building bases
    void prepareMesh()
    {
        buildEdgeSlaveChain();
        buildFaceSlaveChain();
        buildElementVertexAdjacency();
        buildIndexMapping();
    }

    int n_elements = 0;
    int n_verts = 0;

    std::string mesh_path;

    std::vector<ncElem> elements;
    std::vector<ncVert> vertices;
    std::vector<ncBoundary> edges;
    std::vector<ncBoundary> faces;

protected:

    std::vector<int> all2ValidMap, valid2AllMap;
    std::unordered_map< Eigen::Vector2i, int, ArrayHasher2D >  midpointMap;
    std::unordered_map< Eigen::Vector2i, int, ArrayHasher2D >  edgeMap;

    std::vector<int> all2ValidVertexMap, valid2AllVertexMap;

    std::vector<int> refineHistory;

    // elementAdj(i, j) = 1 iff element i touches element j
    Eigen::SparseMatrix<bool, Eigen::RowMajor> elementAdj;
};

}

#endif

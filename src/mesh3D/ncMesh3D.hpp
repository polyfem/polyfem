#ifndef NCMESH3D_HPP
#define NCMESH3D_HPP

#include "polyfem/ncMesh.hpp"
#include <set>
#include <tuple>

// inline int Hash(int p1, int p2) const
// { return (984120265*p1 + 125965121*p2); }

// inline int Hash(int p1, int p2, int p3) const
// { return (984120265*p1 + 125965121*p2 + 495698413*p3); }

namespace polyfem
{

class ncMesh3D: public ncMesh
{
public:
    ncMesh3D(const Eigen::MatrixXd& v, const Eigen::MatrixXi& f) { init(v, f); };
    ncMesh3D(const std::string& path) { init(path); };
    ~ncMesh3D() {};

    int dim() const override { return 3; };

    void init(const Eigen::MatrixXd& v, const Eigen::MatrixXi& f) override;
    void init(const std::string& path) override;

    std::shared_ptr<ncMesh> clone() const override
    {
        return std::make_shared<ncMesh3D>(*this);
    }

    void writeMESH(const std::string& path) const;

    // edges are created if not exist
    // return the id of this new element
    int addElement(Eigen::Vector4i v, int parent = -1);
    
    // refine one element
    void refineElement(int id) override;

    // coarsen one element
    void coarsenElement(int id) override;

    // find the face, return -1 if not exists
    int findFace(Eigen::Vector3i v) const override;
    int findFace(const int v1, const int v2, const int v3) const override { return findFace(Eigen::Vector3i(v1, v2, v3)); };
    // find the face, create one if not exists
    int getFace(Eigen::Vector3i v) override;
    int getFace(const int v1, const int v2, const int v3) override { return getFace(Eigen::Vector3i(v1, v2, v3)); };

    const ncBoundary& interface(int id) const override
    {
        assert(id < faces.size() && id >= 0);
        return faces[id];
    };

    int n_interfaces() const override
    {
        return faces.size();
    };

    int findInterface(Eigen::VectorXi v) const override
    {
        assert(v.size() == 3);
        Eigen::Vector3i v_ = v;
        return findFace(v_);
    };

    // get the id of the lowest order element that share this edge
    int getLowestOrderElementOnEdge(const int edge_id) const
    {
        int min_order_elem = edges[edge_id].get_element();
        for (int elem : edges[edge_id].elem_list) {
            if (elements[min_order_elem].order > elements[elem].order)
                min_order_elem = elem;
        }
        return min_order_elem;
    };

    double faceArea(int f) const override;
    double elementVolume(int e) const override;

    // build a partial mesh including a subset of elements
    void buildPartialMesh(const std::vector<int>& elem_list, ncMesh3D& submesh) const;

    // mark the true boundary vertices
    void markBoundary() override;

    // assign the orders to elements and edges, should call prepareMesh() first
    void assignOrders(Eigen::VectorXi& orders) override;
    void assignOrders(const int order) override
    {
        for (int i = 0; i < elements.size(); i++)
            elements[i].order = order;
        
        for (int j = 0; j < edges.size(); j++)
            edges[j].order = order;
        
        for (int k = 0; k < faces.size(); k++)
            faces[k].order = order;
        
        max_order = order;
        min_order = order;
    };
    bool checkOrders() const;

    // compute local coordinates of global points in elements[e]
    void global2Local(const int e, Eigen::MatrixXd& points) const override;
    void global2Local(Eigen::MatrixXd& points, const Eigen::VectorXi& new_id) const;
    // compute global coordinates of local points in elements[e]
    void local2Global(const int e, Eigen::MatrixXd& points) const override;
    void local2Global(Eigen::MatrixXd& points, const Eigen::VectorXi& new_id) const;
    // determine if points are inside elements[e]
    void isInside(const int e, const Eigen::MatrixXd& points, std::vector<bool>& is_inside) const override;
    bool isInside(const int e, const Eigen::MatrixXd& point) const override;

    void global2LocalEdge(const int edge, const Eigen::MatrixXd& points, Eigen::MatrixXd& uv) const;
    void global2LocalFace(const int face, const Eigen::MatrixXd& points, Eigen::MatrixXd& uv) const;
    void local2GlobalEdge(const int edge, const Eigen::MatrixXd& uv, Eigen::MatrixXd& points) const;
    void local2GlobalFace(const int face, const Eigen::MatrixXd& uv, Eigen::MatrixXd& points) const;

    // map the weight on face to the barycentric coordinate in element
    static Eigen::Vector3d faceWeight2ElemWeight(const int l, const Eigen::Vector2d& pos);
    // map the barycentric coordinate in element to the weight on face
    static Eigen::Vector2d elemWeight2faceWeight(const int l, const Eigen::Vector3d& pos);

    // map the weight on edge to the barycentric coordinate in element
    static Eigen::Vector3d edgeWeight2ElemWeight(const int l, const double& pos);
    // map the barycentric coordinate in element to the weight on edge
    static double elemWeight2edgeWeight(const int l, const Eigen::Vector3d& pos);

    void nodesOnEdge(const int v1, const int v2, std::set<int>& nodes) const;
    void nodesOnFace(const int v1, const int v2, const int v3, std::set<int>& nodes) const;
    void nodesOnElem(const int e, std::set<int>& nodes) const;
    // build the sparse matrix Adj
    void buildElementAdj(int ring = 1) override;

    // call traverseEdge() for every valid edge, and store everything needed
    void buildEdgeSlaveChain() override;

    // list all slave edges of a potential master face, returns nothing if it's a slave or conforming face
    void traverseFace(int v1, int v2, int v3, Eigen::Vector2d p1, Eigen::Vector2d p2, Eigen::Vector2d p3, int depth, std::vector<slave_face>& face_list, std::vector<int>& edge_list) const;

    // call traverseFace() for every valid face, and store everything needed
    void buildFaceSlaveChain() override;

    // assign ncElement3D.master_edges and ncVertex3D.weight
    void buildElementVertexAdjacency() override;

    int boundary_quadrature(const int ncelem_id, const int n_samples, Eigen::MatrixXd& local_pts, Eigen::MatrixXd& uv, Eigen::MatrixXd& normals, Eigen::VectorXd& weights, Eigen::VectorXi& face_ids) const override;
    int boundary_quadrature(const int ncelem_id, const int ncface_id, const int n_samples, Eigen::MatrixXd& local_pts, Eigen::MatrixXd& normals, Eigen::VectorXd& weights) const override;

protected:
    std::unordered_map< Eigen::Vector3i, int, ArrayHasher3D >  faceMap;
};

} // namespace polyfem

#endif

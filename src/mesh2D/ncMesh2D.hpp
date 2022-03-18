#ifndef NCMESH2D_HPP
#define NCMESH2D_HPP

#include "polyfem/ncMesh.hpp"

// inline int Hash(int p1, int p2) const
// { return (984120265*p1 + 125965121*p2); }

// inline int Hash(int p1, int p2, int p3) const
// { return (984120265*p1 + 125965121*p2 + 495698413*p3); }

namespace polyfem
{

class ncMesh2D: public ncMesh
{
public:
    ncMesh2D(const Eigen::MatrixXd& v, const Eigen::MatrixXi& f) { init(v, f); };
    ncMesh2D(const std::string& path) { init(path); };
    ~ncMesh2D() {};

    int dim() const override { return 2; };

    void init(const Eigen::MatrixXd& v, const Eigen::MatrixXi& f) override;
    void init(const std::string& path) override;

    std::shared_ptr<ncMesh> clone() const override
    {
        return std::make_shared<ncMesh2D>(*this);
    }

    void writeOBJ(const std::string& path) const;

    // edges are created if not exist
    // return the id of this new element
    int addElement(Eigen::Vector3i v, int parent = -1);
    
    // refine one element
    virtual void refineElement(int id) override;

    // coarsen one element
    void coarsenElement(int id) override;

    const ncBoundary& interface(int id) const override
    {
        assert(id < edges.size() && id >= 0);
        return edges[id];
    };

    int n_interfaces() const override
    {
        return edges.size();
    };

    int findInterface(Eigen::VectorXi v) const override
    {
        assert(v.size() == 2);
        Eigen::Vector2i v_ = v;
        return findEdge(v_);
    };

    // find the local index of edges[l] in the elements[e]
    int globalEdge2LocalEdge(const int e, const int l) const;

    double elementArea(int e) const override;

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

        max_order = order;
        min_order = order;
    };

    // compute local coordinates of global points in elements[e]
    void global2Local(const int e, Eigen::MatrixXd& points) const override;
    // compute global coordinates of local points in elements[e]
    void local2Global(const int e, Eigen::MatrixXd& points) const override;
    // determine if points are inside elements[e]
    void isInside(const int e, const Eigen::MatrixXd& points, std::vector<bool>& is_inside) const override;
    bool isInside(const int e, const Eigen::MatrixXd& point) const override;

    // map the weight on edge to the barycentric coordinate in element
    static Eigen::Vector2d edgeWeight2ElemWeight(const int l, const double w);
    // map the barycentric coordinate in element to the weight on edge
    static double elemWeight2EdgeWeight(const int l, const Eigen::Vector2d& pos);

    // build the sparse matrix Adj
    void buildElementAdj(int ring = 1) override;

    // call traverseEdge() for every interface, and store everything needed
    void buildEdgeSlaveChain() override;

    // assign ncElement2D.master_edges and ncVertex2D.weight
    void buildElementVertexAdjacency() override;

    int boundary_quadrature(const int ncelem_id, const int n_samples, Eigen::MatrixXd& local_pts, Eigen::MatrixXd& uv, Eigen::MatrixXd& normals, Eigen::VectorXd& weights, Eigen::VectorXi& face_ids) const override;
    int boundary_quadrature(const int ncelem_id, const int ncface_id, const int n_samples, Eigen::MatrixXd& local_pts, Eigen::MatrixXd& normals, Eigen::VectorXd& weights) const override;
};

} // namespace polyfem

#endif

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

    // edges are created if not exist
    // return the id of this new element
    int addElement(Eigen::Vector3i v, int parent = -1);
    
    // refine one element
    virtual void refineElement(int id) override;

    // coarsen one element
    void coarsenElement(int id) override;

    // find the local index of edges[l] in the elements[e]
    int globalEdge2LocalEdge(const int e, const int l) const;

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
};

} // namespace polyfem

#endif

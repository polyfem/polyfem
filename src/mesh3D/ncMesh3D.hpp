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

    // build a partial mesh including a subset of elements
    void buildPartialMesh(const std::vector<int>& elem_list, ncMesh3D& submesh) const;

    // mark the true boundary vertices
    void markBoundary() override;

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

protected:
    std::unordered_map< Eigen::Vector3i, int, ArrayHasher3D >  faceMap;

private:
    void nodesOnEdge(const int v1, const int v2, std::set<int>& nodes) const;
    void nodesOnFace(const int v1, const int v2, const int v3, std::set<int>& nodes) const;
    void nodesOnElem(const int e, std::set<int>& nodes) const;
};

} // namespace polyfem

#endif

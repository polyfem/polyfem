#include "ncMesh2D.hpp"
#include <stdlib.h>
#include <algorithm>
#include <igl/readOBJ.h>
#include <igl/readMSH.h>
#include <boost/algorithm/string.hpp>
#include <polyfem/auto_p_bases.hpp>
#include <polyfem/LineQuadrature.hpp>
#include <polyfem/par_for.hpp>
#include <polyfem/MaybeParallelFor.hpp>
#include <polyfem/Logger.hpp>

#ifdef POLYFEM_WITH_TBB
#include <tbb/parallel_for.h>
#endif

namespace polyfem
{

void ncMesh2D::init(const Eigen::MatrixXd& v, const Eigen::MatrixXi& f)
{
    n_elements = 0; 
    vertices.reserve(v.rows());
    for (int i = 0; i < v.rows(); i++) {
        vertices.emplace_back(v.row(i));
    }
    for (int i = 0; i < f.rows(); i++) {
        addElement(f.row(i), -1);
    }
}

void ncMesh2D::init(const std::string& path)
{
    Eigen::MatrixXd v_;
    Eigen::MatrixXd v;
    Eigen::MatrixXi f;
    if (boost::algorithm::ends_with(path, "obj"))
        igl::readOBJ(path, v_, f);
    else if (boost::algorithm::ends_with(path, "msh"))
        igl::readMSH(path, v_, f);

    v.resize(v_.rows(), 2);
    v = v_.block(0, 0, v.rows(), v.cols());

    for (int i = 0; i < f.rows(); i++) {
        auto e1 = v.row(f(i, 1)) - v.row(f(i, 0));
        auto e2 = v.row(f(i, 2)) - v.row(f(i, 0));
        if (e1(0) * e2(1) - e2(0) * e1(1) < 0) {
            int tmp = f(i, 0);
            f(i, 0) = f(i, 1);
            f(i, 1) = tmp;
        }
    }

    init(v, f);
    mesh_path = path;
}

int ncMesh2D::addElement(Eigen::Vector3i v, int parent)
{
    const int id = elements.size();
    const int level = (parent < 0) ? 0 : elements[parent].level + 1;
    elements.emplace_back(2, v, level, parent);

    for (int i = 0; i < v.size(); i++)
        vertices[v(i)].n_elem++;

    // add edges if not exist
    int edge01 = getEdge(Eigen::Vector2i(v[0],v[1]));
    int edge12 = getEdge(Eigen::Vector2i(v[2],v[1]));
    int edge20 = getEdge(Eigen::Vector2i(v[0],v[2]));

    elements[id].edges << edge01, edge12, edge20;

    edges[edge01].add_element(id);
    edges[edge12].add_element(id);
    edges[edge20].add_element(id);

    n_elements++;

    return id;
}

void ncMesh2D::refineElement(int id)
{
    const auto v = elements[id].vertices;

    assert(elements[id].is_valid() && "Invalid element in refining!");
    elements[id].is_refined = true;
    n_elements--;

    // remove the old element from edge reference
    for (int e = 0; e < 3; e++)
        edges[elements[id].edges(e)].remove_element(id);

    for (int i = 0; i < v.size(); i++)
        vertices[v(i)].n_elem--;

    if (elements[id].children(0) >= 0) {
        for (int c = 0; c < elements[id].children.size(); c++) {
            auto& elem = elements[elements[id].children(c)];
            elem.is_ghost = false;
            n_elements++;
            for (int le = 0; le < elem.edges.size(); le++)
                edges[elem.edges(le)].add_element(elements[id].children(c));
            for (int i = 0; i < elem.vertices.size(); i++)
                vertices[elem.vertices(i)].n_elem++;
        }
    }
    else {
        // create mid-points if not exist
        const int v01 = getVertex(Eigen::Vector2i(v[0],v[1]));
        const int v12 = getVertex(Eigen::Vector2i(v[2],v[1]));
        const int v20 = getVertex(Eigen::Vector2i(v[0],v[2]));

        // inherite line singularity flag from parent edge
        for (int i = 0; i < v.size(); i++)
            for (int j = 0; j < i; j++) {
                int mid_id = findVertex(v[i], v[j]);
                int edge_id = findEdge(v[i], v[j]);
                int edge1 = getEdge(v[i], mid_id);
                int edge2 = getEdge(v[j], mid_id);
                edges[edge1].flag = edges[edge_id].flag;
                edges[edge2].flag = edges[edge_id].flag;
                edges[edge1].boundary_flag = edges[edge_id].boundary_flag;
                edges[edge2].boundary_flag = edges[edge_id].boundary_flag;
                vertices[mid_id].flag = edges[edge_id].flag;
            }

        // create and insert child elements
        elements[id].children(0) = elements.size(); addElement(Eigen::Vector3i(v[0],v01,v20), id);
        elements[id].children(1) = elements.size(); addElement(Eigen::Vector3i(v[1],v12,v01), id);
        elements[id].children(2) = elements.size(); addElement(Eigen::Vector3i(v[2],v20,v12), id);
        elements[id].children(3) = elements.size(); addElement(Eigen::Vector3i(v12 ,v20,v01), id);
    }
    
    for (int i = 0; i < elements[id].children.size(); i++)
        elements[elements[id].children(i)].order = elements[id].order;

    refineHistory.push_back(id);
}

void ncMesh2D::coarsenElement(int id)
{
    const int parent_id = elements[id].parent;
    auto& parent = elements[parent_id];

    for (int i = 0; i < parent.children.size(); i++)
        assert(elements[parent.children(i)].is_valid() && "Invalid siblings in coarsening!");

    // remove elements
    for (int i = 0; i < parent.children.size(); i++) {
        auto& elem = elements[parent.children(i)];
        elem.is_ghost = true;
        parent.order = std::max(parent.order, elem.order);
        n_elements--;
        for (int le = 0; le < elem.edges.size(); le++)
            edges[elem.edges(le)].remove_element(parent.children(i));
        for (int v = 0; v < elem.vertices.size(); v++)
            vertices[elem.vertices(v)].n_elem--;
    }

    // add element
    parent.is_refined = false;
    n_elements++;
    for (int le = 0; le < parent.edges.size(); le++)
        edges[parent.edges(le)].add_element(parent_id);
    for (int v = 0; v < parent.vertices.size(); v++)
        vertices[parent.vertices(v)].n_elem++;

    refineHistory.push_back(parent_id);
}

int ncMesh2D::globalEdge2LocalEdge(const int e, const int l) const
{
    for (int i = 0; i < 3; i++) {
        if (elements[e].edges[i] == l)
            return i;
    }
    assert(false);
    return 0;
}

int find(const Eigen::Vector3i& vec, int x)
{
    for (int i = 0; i < 3; i++) {
        if (x == vec[i])
            return i;
    }
    return -1;
}

void ncMesh2D::buildEdgeSlaveChain()
{
    for (auto& edge : edges) {
        edge.master = -1;
        edge.slaves.clear();
        edge.weights.setConstant(-1);
    }

    Eigen::Vector2i v;
    std::vector<slave_edge> slaves;
    for (int e_id = 0; e_id < elements.size(); e_id++) {
        const auto& element = elements[e_id];
        if (element.is_not_valid())
            continue;
        for (int edge_local = 0; edge_local < 3; edge_local++) {
            v << element.vertices[edge_local], element.vertices[(edge_local+1)%3];  // order is important here!
            int edge_global = element.edges[edge_local];
            assert(edge_global >= 0);
            traverseEdge(v, 0, 1, 0, slaves);
            for (auto& s : slaves) {
                edges[s.id].master = edge_global;
                edges[edge_global].slaves.push_back(s.id);
                edges[s.id].weights << s.p1, s.p2;
            }
            slaves.clear();
        }
    }
}

void ncMesh2D::assignOrders(Eigen::VectorXi& orders)
{
    assert(orders.size() == n_elements);

    for (auto& edge : edges) {
        edge.order = 100;
    }

    // assign orders to elements
    for (int i = 0; i < n_elements; i++)
		elements[valid2All(i)].order = orders[i];


    // assign edge orders
    for (int i = 0; i < n_elements; i++) {
		auto &elem = elements[valid2All(i)];
        for (int j = 0; j < 3; j++) {
            auto& edge = edges[elem.edges[j]];
            if (edge.order > elem.order)
                edge.order = elem.order;
        }
    }
    // for every master edge, reduce the order to the min of its slave edge orders
	for (auto& small_edge : edges)
	{
		int large_edge_id = small_edge.master;
		if (large_edge_id < 0)
			continue;
		auto &large_edge = edges[large_edge_id];
		large_edge.order = std::min(small_edge.order, large_edge.order);
	}
    // for every slave edge, reduce the order to its master edge order
	for (auto &small_edge : edges)
	{
		int large_edge_id = small_edge.master;
		if (large_edge_id < 0)
			continue;
		auto &large_edge = edges[large_edge_id];
		small_edge.order = std::min(small_edge.order, large_edge.order);
	}

    min_order = orders.minCoeff();
    max_order = orders.maxCoeff();
}

void ncMesh2D::markBoundary()
{
    for (auto& edge : edges) {
        if (edge.n_elem() == 1)
            edge.isboundary = true;
        else
            edge.isboundary = false;
    }

    for (auto& edge : edges) {
        if (edge.master >= 0) {
            edge.isboundary = false;
            edges[edge.master].isboundary = false;
        }
    }

    for (auto& vert : vertices)
        vert.isboundary = false;

    for (auto& edge : edges) {
        if (edge.isboundary && edge.n_elem()) {
            for (int j = 0; j < 2; j++)
                vertices[edge.vertices(j)].isboundary = true;
        }
    }
}

double line_weight(Eigen::Matrix<double, 2, 2>& e, Eigen::VectorXd& v)
{
    assert(v.size() == 2);
    double w1 = (v(0) - e(0, 0)) / (e(1, 0) - e(0, 0));
    double w2 = (v(1) - e(0, 1)) / (e(1, 1) - e(0, 1));
    if (0 <= w1 && w1 <= 1)
        return w1;
    else
        return w2;
}

void ncMesh2D::buildElementVertexAdjacency()
{
    for (auto& vert : vertices) {
        vert.edge = -1;
        vert.weight = -1;
    }
    
    Eigen::VectorXi vertexEdgeAdjacency;
    vertexEdgeAdjacency.setConstant(vertices.size(), 1, -1);

    for (int small_edge = 0; small_edge < edges.size(); small_edge++) {
        if (edges[small_edge].n_elem() == 0)
            continue;

        int large_edge = edges[small_edge].master;
        if (large_edge < 0)
            continue;
        
        int large_elem = edges[large_edge].get_element();
        for (int j = 0; j < 2; j++) {
            int v_id = edges[small_edge].vertices(j);
            if (find(elements[large_elem].vertices, v_id) < 0) // or maybe 0 < weights(large_edge, v_id) < 1
                vertexEdgeAdjacency[v_id] = large_edge;
        }
    }
    
    for (auto& element : elements) {
        if (element.is_not_valid())
            continue;
        for (int v_local = 0; v_local < 3; v_local++) {
            int v_global = element.vertices[v_local];
            if (vertexEdgeAdjacency[v_global] < 0)
                continue;
            
            auto& large_edge = edges[vertexEdgeAdjacency[v_global]];
            auto& large_element = elements[large_edge.get_element()];
            vertices[v_global].edge = vertexEdgeAdjacency[v_global];

            int e_local = find(large_element.edges, vertices[v_global].edge);
            Eigen::Matrix<double, 2, 2> edge;
            edge.row(0) = vertices[large_element.vertices[e_local]].pos;
            edge.row(1) = vertices[large_element.vertices[(e_local+1) % 3]].pos;
            vertices[v_global].weight = line_weight(edge, vertices[v_global].pos);
        }
    }
}

void ncMesh2D::buildElementAdj(int ring)
{
    assert(ring > 0);
    Eigen::SparseMatrix<bool> vertElemAdj(vertices.size(), n_elements);  // vertElemAdj(i, j) = 1 iff vert i touches elem j
    std::vector<Eigen::Triplet<bool> > triplets;

    // regular vertices
    for (int e = 0; e < n_elements; e++) {
        const auto& elem = elements[valid2All(e)];
        if (elem.is_not_valid())
            continue;

        for (int v = 0; v < 3; v++)
            triplets.emplace_back(elem.vertices(v), e, true);
    }

    // hanging vertices
    for (int v = 0; v < vertices.size(); v++) {
        const auto& vert = vertices[v];
        if (vert.edge < 0)
            continue;

        triplets.emplace_back(v, all2Valid(edges[vert.edge].get_element()), true);
    }

    vertElemAdj.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SparseMatrix<bool> tmp = Eigen::SparseMatrix<bool>(vertElemAdj.transpose()) * vertElemAdj;
    elementAdj = tmp;
    while (ring > 1) {
        elementAdj = elementAdj * tmp;
        ring--;
    }
}

Eigen::Vector2d ncMesh2D::edgeWeight2ElemWeight(const int l, const double w)
{
    Eigen::Vector2d pos;
    switch(l) {
        case 0:
            pos(0) = w;
            pos(1) = 0;
            break;
        case 1:
            pos(0) = 1 - w;
            pos(1) = w;
            break;
        case 2:
            pos(0) = 0;
            pos(1) = 1 - w;
            break;
        default:
            assert(false);
    }
    return pos;
}

double ncMesh2D::elemWeight2EdgeWeight(const int l, const Eigen::Vector2d& pos)
{
    double w = -1;
    switch (l) {
        case 0:
            w = pos(0);
            assert(fabs(pos(1)) < 1e-12);
            break;
        case 1:
            w = pos(1);
            assert(fabs(pos(0)+pos(1)-1) < 1e-12);
            break;
        case 2:
            w = 1 - pos(1);
            assert(fabs(pos(0)) < 1e-12);
            break;
        default:
            assert(false);
    }
    return w;
}

}

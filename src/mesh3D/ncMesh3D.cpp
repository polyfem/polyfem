#include "ncMesh3D.hpp"
#include <stdlib.h>
#include <algorithm>
#include <igl/writeMESH.h>
#include <igl/readMSH.h>
#include <polyfem/auto_p_bases.hpp>
#include <polyfem/TriQuadrature.hpp>
#include <polyfem/Logger.hpp>

#ifdef POLYFEM_WITH_TBB
#include <tbb/parallel_for.h>
#endif

#include <queue>

namespace polyfem
{

void ncMesh3D::init(const Eigen::MatrixXd& v, const Eigen::MatrixXi& f)
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

void ncMesh3D::init(const std::string& path)
{
    assert(path.substr(path.find_last_of(".") + 1) == "msh");
    Eigen::MatrixXd v;
    Eigen::MatrixXi tri, tet;
    Eigen::VectorXi triTag, tetTag;
    igl::readMSH(path, v, tri, tet, triTag, tetTag);

    init(v, tet);
    mesh_path = path;
}

void ncMesh3D::writeMESH(const std::string& path) const
{
    assert(path.substr(path.find_last_of(".") + 1) == "mesh");
    Eigen::MatrixXd v;
    Eigen::MatrixXi tet;
    compress(v, tet);
    igl::writeMESH(path, v, tet, Eigen::MatrixXi());
}

int ncMesh3D::findFace(Eigen::Vector3i v) const
{
    std::sort(v.data(), v.data()+v.size());
    auto search = faceMap.find(v);
    if (search != faceMap.end())
        return search->second;
    else
        return -1;
}

int ncMesh3D::getFace(Eigen::Vector3i v)
{
    std::sort(v.data(), v.data()+v.size());
    int id = findFace(v);
    if (id < 0) {
        faces.emplace_back(v);
        id = faces.size() - 1;
        faceMap.emplace(v, id);
    }
    return id;
}

int ncMesh3D::addElement(Eigen::Vector4i v, int parent)
{
    const int id = elements.size();
    const int level = (parent < 0) ? 0 : elements[parent].level + 1;
    elements.emplace_back(3, v, level, parent);

    // add faces if not exist
    const int face012 = getFace(v[0],v[1],v[2]);
    const int face013 = getFace(v[0],v[1],v[3]);
    const int face123 = getFace(v[1],v[2],v[3]);
    const int face203 = getFace(v[0],v[2],v[3]);

    faces[face012].add_element(id);
    faces[face013].add_element(id);
    faces[face123].add_element(id);
    faces[face203].add_element(id);

    elements[id].faces << face012, face013, face123, face203;

    // add edges if not exist
    const int edge01 = getEdge(v[0],v[1]);
    const int edge12 = getEdge(v[2],v[1]);
    const int edge20 = getEdge(v[0],v[2]);
    const int edge03 = getEdge(v[0],v[3]);
    const int edge13 = getEdge(v[3],v[1]);
    const int edge23 = getEdge(v[2],v[3]);

    edges[edge01].add_element(id);
    edges[edge12].add_element(id);
    edges[edge20].add_element(id);
    edges[edge03].add_element(id);
    edges[edge13].add_element(id);
    edges[edge23].add_element(id);

    elements[id].edges << edge01, edge12, edge20, edge03, edge13, edge23;

    n_elements++;

    for (int i = 0; i < v.size(); i++)
        vertices[v[i]].n_elem++;

    return id;
}

/*
A refined tet element for reference

$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
10
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 0.5 0 0
6 0 0.5 0
7 0 0 0.5
8 0.5 0.5 0
9 0.5 0 0.5
10 0 0.5 0.5
$EndNodes
$Elements
32
1 1 2 0 1 1 5
2 1 2 0 1 5 2
3 1 2 0 1 1 6
4 1 2 0 1 6 3
5 1 2 0 1 1 7
6 1 2 0 1 7 4
7 1 2 0 1 2 8
8 1 2 0 1 8 3
9 2 2 0 1 1 5 6
10 2 2 0 1 5 8 6
11 2 2 0 1 5 2 8
12 2 2 0 1 6 8 3
13 2 2 0 1 1 5 7
14 2 2 0 1 5 9 7
15 2 2 0 1 5 2 9
16 2 2 0 1 7 9 4
17 2 2 0 1 1 6 7
18 2 2 0 1 6 10 7
19 2 2 0 1 6 3 10
20 2 2 0 1 7 10 4
21 2 2 0 1 2 8 9
22 2 2 0 1 8 10 9
23 2 2 0 1 8 3 10
24 2 2 0 1 9 10 4
25 4 2 0 1 1 5 6 7
26 4 2 0 1 5 2 8 9
27 4 2 0 1 6 8 3 10
28 4 2 0 1 7 9 10 4
29 4 2 0 1 5 6 7 9
30 4 2 0 1 5 9 8 6
31 4 2 0 1 6 7 9 10
32 4 2 0 1 6 10 9 8
$EndElements
*/

void ncMesh3D::refineElement(int id)
{
    const auto v = elements[id].geom_vertices;

    assert(elements[id].is_valid());
    elements[id].is_refined = true;
    n_elements--;

    for (int f = 0; f < elements[id].faces.size(); f++)
        faces[elements[id].faces(f)].remove_element(id);

    for (int e = 0; e < elements[id].edges.size(); e++)
        edges[elements[id].edges(e)].remove_element(id);

    for (int i = 0; i < v.size(); i++)
        vertices[v(i)].n_elem--;

    if (elements[id].children(0) >= 0) {
        for (int c = 0; c < elements[id].children.size(); c++) {
            const int child_id = elements[id].children(c);
            auto& elem = elements[child_id];
            elem.is_ghost = false;
            n_elements++;

            for (int f = 0; f < elem.faces.size(); f++)
                faces[elem.faces(f)].add_element(child_id);

            for (int e = 0; e < elem.edges.size(); e++)
                edges[elem.edges(e)].add_element(child_id);

            for (int v = 0; v < elem.vertices.size(); v++)
                vertices[elem.vertices(v)].n_elem++;
        }
    }
    else {
        // create mid-points if not exist
        const int v1 = v[0];
        const int v2 = v[1];
        const int v3 = v[2];
        const int v4 = v[3];
        const int v5 = getVertex(Eigen::Vector2i(v1,v2));
        const int v8 = getVertex(Eigen::Vector2i(v3,v2));
        const int v6 = getVertex(Eigen::Vector2i(v1,v3));
        const int v7 = getVertex(Eigen::Vector2i(v1,v4));
        const int v9 = getVertex(Eigen::Vector2i(v4,v2));
        const int v10 = getVertex(Eigen::Vector2i(v3,v4));

        // inherite line singularity flag from parent edge
        for (int i = 0; i < v.size(); i++)
            for (int j = 0; j < i; j++) {
                int mid_id = findVertex(v[i], v[j]);
                int edge_id = findEdge(v[i], v[j]);
                int edge1 = getEdge(v[i], mid_id);
                int edge2 = getEdge(v[j], mid_id);
                edges[edge1].flag = edges[edge_id].flag;
                edges[edge2].flag = edges[edge_id].flag;
                vertices[mid_id].flag = edges[edge_id].flag;
            }

        // create children
        elements[id].children(0) = elements.size(); addElement(Eigen::Vector4i(v1,v5,v6,v7), id);
        elements[id].children(1) = elements.size(); addElement(Eigen::Vector4i(v5,v2,v8,v9), id);
        elements[id].children(2) = elements.size(); addElement(Eigen::Vector4i(v6,v8,v3,v10), id);
        elements[id].children(3) = elements.size(); addElement(Eigen::Vector4i(v7,v9,v10,v4), id);
        elements[id].children(4) = elements.size(); addElement(Eigen::Vector4i(v5,v6,v7,v9), id);
        elements[id].children(5) = elements.size(); addElement(Eigen::Vector4i(v5,v9,v8,v6), id);
        elements[id].children(6) = elements.size(); addElement(Eigen::Vector4i(v6,v7,v9,v10), id);
        elements[id].children(7) = elements.size(); addElement(Eigen::Vector4i(v6,v10,v9,v8), id);
    }

    for (int i = 0; i < elements[id].children.size(); i++)
        elements[elements[id].children(i)].order = elements[id].order;

    refineHistory.push_back(id);
}

void ncMesh3D::coarsenElement(int id)
{
    const int parent_id = elements[id].parent;
    auto& parent = elements[parent_id];

    for (int i = 0; i < parent.children.size(); i++)
        assert(elements[parent.children(i)].is_valid() && "Invalid siblings in coarsening!");

    // remove elements
    for (int c = 0; c < parent.children.size(); c++) {
        auto& elem = elements[parent.children(c)];
        elem.is_ghost = true;
        parent.order = std::max(parent.order, elem.order);
        n_elements--;

        for (int f = 0; f < elem.faces.size(); f++)
            faces[elem.faces(f)].remove_element(parent.children(c));

        for (int e = 0; e < elem.edges.size(); e++)
            edges[elem.edges(e)].remove_element(parent.children(c));

        for (int v = 0; v < elem.vertices.size(); v++)
            vertices[elem.vertices(v)].n_elem--;
    }

    // add element
    parent.is_refined = false;
    n_elements++;

    for (int f = 0; f < parent.faces.size(); f++)
        faces[parent.faces(f)].add_element(parent_id);

    for (int e = 0; e < parent.edges.size(); e++)
        edges[parent.edges(e)].add_element(parent_id);

    for (int v = 0; v < parent.vertices.size(); v++)
        vertices[parent.vertices(v)].n_elem++;

    refineHistory.push_back(parent_id);
}

// int ncMesh3D::globalFace2LocalFace(const int e, const int l) const
// {
//     for (int i = 0; i < elements[e].faces.size(); i++) {
//         if (elements[e].faces[i] == l)
//             return i;
//     }
//     assert(false);
// }

// int ncMesh3D::globalEdge2LocalEdge(const int e, const int l) const
// {
//     for (int i = 0; i < elements[e].edges.size(); i++) {
//         if (elements[e].edges[i] == l)
//             return i;
//     }
//     assert(false);
// }

double ncMesh3D::faceArea(int f) const
{
    assert(f >= 0 && f < faces.size());

    Eigen::Vector3d x, y;
    x = vertices[faces[f].vertices(1)].pos - vertices[faces[f].vertices(0)].pos;
    y = vertices[faces[f].vertices(2)].pos - vertices[faces[f].vertices(0)].pos;

    return (x.cross(y)).norm() / 2;
};

double ncMesh3D::elementVolume(int e) const
{
    Eigen::Matrix3d J;
    for (int d = 0; d < 3; d++)
        J.col(d) = vertices[elements[e].vertices(d + 1)].pos - vertices[elements[e].vertices(0)].pos;

    return std::abs(J.determinant()) / 6;
}

void ncMesh3D::buildPartialMesh(const std::vector<int>& elem_list, ncMesh3D& submesh) const
{
    int min_level = 100;
    for (int i : elem_list) {
        if (elements[i].level < min_level)
            min_level = elements[i].level;
    }

    std::set<int> large_elements;
    std::vector<int> list;
    for (int i : elem_list) {
        while(elements[i].level > min_level)
            i = elements[i].parent;
        large_elements.insert(i);
    }
    list.clear();
    std::copy(large_elements.begin(), large_elements.end(), std::back_inserter(list));

    Eigen::MatrixXi f;
    Eigen::MatrixXd v;
    ncMesh::buildPartialMesh(list, v, f);
    submesh.init(v, f);

    igl::writeMESH("debug.mesh", v, f, Eigen::MatrixXi());

    std::vector<int> globalElemId2Local(elements.size(), -1);
    for (int i = 0; i < list.size(); i++) {
        globalElemId2Local[list[i]] = i;
    }

    std::queue<int> children_queue;
    for (int x : list)
        children_queue.push(x);
    while(!children_queue.empty()) {
        int x = children_queue.front(); children_queue.pop();
        if (!elements[x].is_refined) {
            submesh.elements[globalElemId2Local[x]].order = elements[x].order;
            continue;
        }
        submesh.refineElement(globalElemId2Local[x]);
        for (int i = 0; i < submesh.elements[globalElemId2Local[x]].children.size(); i++) {
            globalElemId2Local[elements[x].children(i)] = submesh.elements[globalElemId2Local[x]].children(i);
            children_queue.push(elements[x].children(i));
        }
    }
}

void ncMesh3D::markBoundary()
{
    for (auto& face : faces) {
        if (face.n_elem() == 1)
            face.isboundary = true;
        else
            face.isboundary = false;
    }

    for (auto& face : faces) {
        if (face.master >= 0) {
            face.isboundary = false;
            faces[face.master].isboundary = false;
        }
    }

    for (auto& vert : vertices)
        vert.isboundary = false;

    for (auto& edge : edges)
        edge.isboundary = false;

    for (auto& face : faces) {
        if (face.isboundary && face.n_elem()) {
            for (int j = 0; j < 3; j++) {
                vertices[face.vertices(j)].isboundary = true;
                for (int i = 0; i < j; i++) {
                    const int edge_id = findEdge(face.vertices(i), face.vertices(j));
                    assert(edge_id >= 0);
                    edges[edge_id].isboundary = true;
                }
            }
        }
    }
}

bool ncMesh3D::checkOrders() const
{
    for (auto& face : faces) {
        if (face.n_elem() == 0)
            continue;
        for (int elem : face.elem_list) {
            if (face.order > elements[elem].order) return false;
        }

        // for (int i = 0; i < face.vertices.size(); i++)
        //     for (int j = 0; j < i; j++)
        //         if (face.order > edges[findEdge(face.vertices(i),face.vertices(j))].order) return false;
    }

    for (auto& edge : edges) {
        if (edge.master_face >= 0 && edge.master < 0)
            if (faces[edge.master_face].order > edge.order) return false;
    }

	for (auto& small_face : faces) {
		int large_face_id = small_face.master;
		if (large_face_id < 0)
			continue;
		auto &large_face = faces[large_face_id];
		const auto &small_element = elements[small_face.get_element()];
        if (small_face.order != large_face.order) return false;
	}

    for (auto &face : faces) {
        if (face.n_elem() == 0)
            continue;
        for (int i = 0; i < 3; i++) {
            auto& edge = edges[findEdge(face.vertices(i),face.vertices((i+1)%3))];
            if (edge.order > face.order) return false;
        }
    }

	for (auto& small_edge : edges) {
		int large_edge_id = small_edge.master;
		if (large_edge_id < 0)
			continue;
		auto &large_edge = edges[large_edge_id];
		if (large_edge.order != small_edge.order) return false;
	}

    return true;
}

void ncMesh3D::assignOrders(Eigen::VectorXi& orders)
{
    assert(orders.size() == n_elements);
    
    for (auto& edge : edges)
        edge.order = 100;

    for (auto& face : faces)
        face.order = 100;

    // assign orders to elements
    for (int i = 0; i < n_elements; i++)
        elements[valid2All(i)].order = orders[i];
    
    // assign face orders
    for (auto& face : faces) {
        if (face.n_elem() == 0)
            continue;
        for (int elem : face.elem_list) {
            face.order = std::min(face.order, elements[elem].order);
        }
    }

    do {
        // for every master face, reduce the order to the min of its slave face orders
        for (auto& small_face : faces) {
            int large_face_id = small_face.master;
            if (large_face_id < 0)
                continue;
            auto &large_face = faces[large_face_id];
            const auto &small_element = elements[small_face.get_element()];
            large_face.order = std::min(small_element.order, large_face.order);
        }

        // for every face, reduce the order to the min of its edges
        // for (auto& face : faces) {
        //     for (int i = 0; i < face.vertices.size(); i++)
        //         for (int j = 0; j < i; j++)
        //             face.order = std::min(face.order, edges[findEdge(face.vertices(i),face.vertices(j))].order);
        // }

        // for every slave face, reduce the order to its master face order
        for (auto &small_face : faces) {
            int large_face_id = small_face.master;
            if (large_face_id < 0)
                continue;
            auto &large_face = faces[large_face_id];
            small_face.order = std::min(small_face.order, large_face.order);
        }

        // for every edge, reduce the order to its adjacent face order
        for (auto &face : faces) {
            if (face.n_elem() == 0)
                continue;
            for (int i = 0; i < 3; i++) {
                auto& edge = edges[findEdge(face.vertices(i),face.vertices((i+1)%3))];
                edge.order = std::min(edge.order, face.order);
            }
        }

        // for every master edge, reduce the order to the min of its slave edge orders
        for (auto& small_edge : edges) {
            int large_edge_id = small_edge.master;
            if (large_edge_id < 0)
                continue;
            auto &large_edge = edges[large_edge_id];
            large_edge.order = std::min(small_edge.order, large_edge.order);
        }
        // for every slave edge, reduce the order to its master edge order
        for (auto &small_edge : edges) {
            int large_edge_id = small_edge.master;
            if (large_edge_id < 0)
                continue;
            auto &large_edge = edges[large_edge_id];
            small_edge.order = std::min(small_edge.order, large_edge.order);
        }

        // if an edge lies in the interior of a face, the face order cannot exceed the edge order
        for (auto& edge : edges) {
            if (edge.n_elem() <= 0)
                continue;
            if (edge.master_face >= 0 && edge.master < 0)
                faces[edge.master_face].order = std::min(edge.order, faces[edge.master_face].order);
        }
    } while (!checkOrders());

    min_order = orders.minCoeff();
    max_order = orders.maxCoeff();
}

Eigen::Vector3d ncMesh3D::faceWeight2ElemWeight(const int l, const Eigen::Vector2d& pos)
{
    Eigen::Matrix<double, 4, 3> v;
    v.row(0) << 0, 0, 0;
    v.row(1) << 1, 0, 0;
    v.row(2) << 0, 1, 0;
    v.row(3) << 0, 0, 1;
    Eigen::Matrix<int, 4, 3> fv;
    fv.row(0) << 0, 1, 2;
    fv.row(1) << 0, 1, 3;
    fv.row(2) << 1, 2, 3;
    fv.row(3) << 2, 0, 3;

    return v.row(fv(l, 0)) * (1 - pos(0) - pos(1)) + v.row(fv(l, 1)) * pos(0) + v.row(fv(l, 2)) * pos(1);
}

Eigen::Vector2d ncMesh3D::elemWeight2faceWeight(const int l, const Eigen::Vector3d& pos)
{
    Eigen::Matrix<double, 4, 3> v;
    v.row(0) << 0, 0, 0;
    v.row(1) << 1, 0, 0;
    v.row(2) << 0, 1, 0;
    v.row(3) << 0, 0, 1;
    Eigen::Matrix<int, 4, 4> fv_; // extend fv to fv_ to include all vertices
    fv_.row(0) << 0, 1, 2, 3;
    fv_.row(1) << 0, 1, 3, 2;
    fv_.row(2) << 1, 2, 3, 0;
    fv_.row(3) << 2, 0, 3, 1;

    Eigen::Matrix3d J;
    for (int i = 0; i < 3; i++)
        J.col(i) = v.row(fv_(l, i+1)) - v.row(fv_(l, 0));

    Eigen::Vector3d x = J.colPivHouseholderQr().solve(pos - v.row(fv_(l, 0)).transpose());

    Eigen::Vector2d uv = x.block(0, 0, 2, 1);
    assert(std::abs(x(2)) < 1e-12);
    return uv;
}

Eigen::Vector3d ncMesh3D::edgeWeight2ElemWeight(const int l, const double& pos)
{
    Eigen::Matrix<double, 4, 3> v;
    v.row(0) << 0, 0, 0;
    v.row(1) << 1, 0, 0;
    v.row(2) << 0, 1, 0;
    v.row(3) << 0, 0, 1;
    Eigen::Matrix<int, 6, 2> ev;
    ev.row(0) << 0, 1;
    ev.row(1) << 1, 2;
    ev.row(2) << 2, 0;

    ev.row(3) << 0, 3;
    ev.row(4) << 1, 3;
    ev.row(5) << 2, 3;

    return v.row(ev(l, 0)) * (1 - pos) + v.row(ev(l, 1)) * pos;
}

double ncMesh3D::elemWeight2edgeWeight(const int l, const Eigen::Vector3d& pos)
{
    Eigen::Matrix<double, 4, 3> v;
    v.row(0) << 0, 0, 0;
    v.row(1) << 1, 0, 0;
    v.row(2) << 0, 1, 0;
    v.row(3) << 0, 0, 1;
    Eigen::Matrix<int, 6, 2> ev;
    ev.row(0) << 0, 1;
    ev.row(1) << 1, 2;
    ev.row(2) << 2, 0;

    ev.row(3) << 0, 3;
    ev.row(4) << 1, 3;
    ev.row(5) << 2, 3;

    for (int d = 0; d < 3; d++) {
        double D = v(ev(l, 1), d) - v(ev(l, 0), d);
        if (std::abs(D) > 1e-12)
            return (pos(d) - v(ev(l, 0), d)) / D;
    }
    assert(false);
    return -1;
}

void ncMesh3D::nodesOnEdge(const int v1, const int v2, std::set<int>& nodes) const
{
    const int v_mid = findVertex(v1, v2);
    nodes.insert(v1);
    nodes.insert(v2);
    if (v_mid >= 0) {
        nodesOnEdge(v1, v_mid, nodes);
        nodesOnEdge(v_mid, v2, nodes);
    }
}

void ncMesh3D::nodesOnFace(const int v1, const int v2, const int v3, std::set<int>& nodes) const
{
    const int v12 = findVertex(v1, v2);
    const int v23 = findVertex(v2, v3);
    const int v31 = findVertex(v1, v3);

    nodesOnEdge(v1, v2, nodes);
    nodesOnEdge(v2, v3, nodes);
    nodesOnEdge(v3, v1, nodes);

    if (v12 >= 0 && v23 >= 0 && v31 >= 0) {
        nodesOnFace(v1,  v12, v31, nodes);
        nodesOnFace(v2,  v12, v23, nodes);
        nodesOnFace(v3,  v23, v31, nodes);
        nodesOnFace(v23, v12, v31, nodes);
    }
}

void ncMesh3D::nodesOnElem(const int e, std::set<int>& nodes) const
{
    const auto& v = elements[e].vertices;
    Eigen::Matrix<int, 4, 3> lf;
    lf << 0, 1, 2,
          0, 1, 3,
          1, 2, 3,
          0, 2, 3;
    for (int i = 0; i < lf.rows(); i++)
        nodesOnFace(v[lf(i, 0)],v[lf(i, 1)],v[lf(i, 2)],nodes);
}

void ncMesh3D::buildElementAdj(int ring)
{
    assert(ring > 0);
    Eigen::SparseMatrix<bool> vertElemAdj(vertices.size(), n_elements);  // vertElemAdj(i, j) = 1 iff vert i touches elem j
    std::vector<Eigen::Triplet<bool> > triplets;
    for (int e = 0; e < n_elements; e++) {
        std::set<int> nodes;
        nodesOnElem(valid2All(e), nodes);

        for (int x : nodes)
            triplets.emplace_back(x, e, true);
    }

    vertElemAdj.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SparseMatrix<bool> tmp = Eigen::SparseMatrix<bool>(vertElemAdj.transpose()) * vertElemAdj;
    elementAdj = tmp;
    while (ring > 1) {
        elementAdj = elementAdj * tmp;
        ring--;
    }
}

void ncMesh3D::buildEdgeSlaveChain()
{
    for (auto& edge : edges) {
        edge.master = -1;
        edge.slaves.clear();
        edge.weights.setConstant(-1);
    }

    // Eigen::Vector2i v;
    std::vector<slave_edge> slaves;
    for (int e_id = 0; e_id < edges.size(); e_id++) {
        auto& edge = edges[e_id];
        if (edge.n_elem() == 0)
            continue;
        traverseEdge(edge.vertices, 0, 1, 0, slaves);
        for (auto& s : slaves) {
            edge.slaves.push_back(s.id);
            if (edges[s.id].master >= 0 && std::abs(edges[s.id].weights(1) - edges[s.id].weights(0)) < std::abs(s.p2 - s.p1))
                continue;
            edges[s.id].master = e_id;
            edges[s.id].weights << s.p1, s.p2;
        }
        slaves.clear();
    }

    // In 3d, it's possible for one edge to have both master and slave edges, but we don't care this case.
    for (auto& edge : edges)
        if (edge.master >= 0 && edge.slaves.size())
            edge.slaves.clear();

}

void ncMesh3D::traverseFace(int v1, int v2, int v3, Eigen::Vector2d p1, Eigen::Vector2d p2, Eigen::Vector2d p3, int depth, std::vector<slave_face>& face_list, std::vector<int>& edge_list) const
{
    int v12 = findVertex(Eigen::Vector2i(v1,v2));
    int v23 = findVertex(Eigen::Vector2i(v3,v2));
    int v31 = findVertex(Eigen::Vector2i(v1,v3));
    std::vector<slave_face> list1, list2, list3, list4;
    std::vector<int> list1_, list2_, list3_, list4_;
    if (depth > 0) {
        edge_list.push_back(findEdge(v1, v2));
        edge_list.push_back(findEdge(v3, v2));
        edge_list.push_back(findEdge(v1, v3));
    }
    if (v12 >= 0 && v23 >= 0 && v31 >= 0) {
        auto p12 = (p1+p2)/2, p23=(p2+p3)/2, p31=(p1+p3)/2;
        traverseFace(v1,v12,v31 ,p1,p12,p31 ,depth+1,list1,list1_);
        traverseFace(v12,v2,v23 ,p12,p2,p23 ,depth+1,list2,list2_);
        traverseFace(v31,v23,v3 ,p31,p23,p3 ,depth+1,list3,list3_);
        traverseFace(v12,v23,v31,p12,p23,p31,depth+1,list4,list4_);

        face_list.insert(face_list.end(), std::make_move_iterator(list1.begin()), std::make_move_iterator(list1.end()));
        face_list.insert(face_list.end(), std::make_move_iterator(list2.begin()), std::make_move_iterator(list2.end()));
        face_list.insert(face_list.end(), std::make_move_iterator(list3.begin()), std::make_move_iterator(list3.end()));
        face_list.insert(face_list.end(), std::make_move_iterator(list4.begin()), std::make_move_iterator(list4.end()));

        edge_list.insert(edge_list.end(), std::make_move_iterator(list1_.begin()), std::make_move_iterator(list1_.end()));
        edge_list.insert(edge_list.end(), std::make_move_iterator(list2_.begin()), std::make_move_iterator(list2_.end()));
        edge_list.insert(edge_list.end(), std::make_move_iterator(list3_.begin()), std::make_move_iterator(list3_.end()));
        edge_list.insert(edge_list.end(), std::make_move_iterator(list4_.begin()), std::make_move_iterator(list4_.end()));
    }
    if (depth > 0) {
        int slave_id = findFace(Eigen::Vector3i(v1,v2,v3));
        if (slave_id >= 0 && faces[slave_id].n_elem() > 0)
            face_list.emplace_back(slave_id, p1, p2, p3);
    }
}

void ncMesh3D::buildFaceSlaveChain()
{
    Eigen::Matrix<int, 4, 3> fv;
    fv.row(0) << 0, 1, 2;
    fv.row(1) << 0, 1, 3;
    fv.row(2) << 1, 2, 3;
    fv.row(3) << 2, 0, 3;

    for (auto& face : faces) {
        face.master = -1;
        face.slaves.clear();
    }

    for (auto& edge : edges) {
        edge.master_face = -1;
    }

    std::vector<slave_face> slaves;
    std::vector<int> interior_edges;
    for (int f_id = 0; f_id < faces.size(); f_id++) {
        auto& face = faces[f_id];
        if (face.n_elem() == 0)
            continue;
        traverseFace(face.vertices(0), face.vertices(1), face.vertices(2), Eigen::Vector2d(0, 0), Eigen::Vector2d(1, 0), Eigen::Vector2d(0, 1), 0, slaves, interior_edges); // order is important
        for (auto& s : slaves) {
            faces[s.id].master = f_id;
            face.slaves.push_back(s.id);
        }
        slaves.clear();
        for (int s : interior_edges)
            if (s >= 0 && edges[s].master < 0 && edges[s].n_elem() > 0)
                edges[s].master_face = f_id;
        interior_edges.clear();
    }
}

void ncMesh3D::buildElementVertexAdjacency()
{
    for (auto& vert : vertices) {
        vert.edge = -1;
        // vert.edge_weight = -1;
        vert.face = -1;
        // vert.face_weight.setConstant(-1);
    }

    // Eigen::VectorXi vertexEdgeAdjacency;
    // vertexEdgeAdjacency.setConstant(vertices.size(), 1, -1);

    for (auto& small_edge : edges) {
        // invalid edges
        if (small_edge.n_elem() == 0)
            continue;

        // not slave edges
        int large_edge = small_edge.master;
        if (large_edge < 0)
            continue;
        assert(edges[large_edge].master < 0);
        
        // slave edges
        for (int j = 0; j < 2; j++) {
            const int v_id = small_edge.vertices(j);
            // hanging nodes
            if (v_id != edges[large_edge].vertices(0) && v_id != edges[large_edge].vertices(1))
                    vertices[v_id].edge = large_edge;
        }
    }

    for (auto& small_face : faces) {
        // invalid faces
        if (small_face.n_elem() == 0)
            continue;

        // not slave faces
        int large_face = small_face.master;
        if (large_face < 0)
            continue;
        
        // slave faces
        for (int j = 0; j < 3; j++) {
            const int v_id = small_face.vertices(j);
            // hanging nodes
            if (v_id != faces[large_face].vertices(0) && v_id != faces[large_face].vertices(1) && v_id != faces[large_face].vertices(2))
                    vertices[v_id].face = large_face;
        }
    }
}

}

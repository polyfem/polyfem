#include "ncMesh.hpp"
#include <polyfem/Logger.hpp>
#include <algorithm>
#include <stdlib.h>
#include <fstream>

namespace polyfem
{

void ncMesh::normalize()
{
    Eigen::VectorXd pos_min = vertices[0].pos, pos_max = vertices[0].pos;
    for (const auto& v : vertices) {
        for (int d = 0; d < dim(); d++) {
            if (v.pos[d] < pos_min[d]) pos_min[d] = v.pos[d];
            if (v.pos[d] > pos_max[d]) pos_max[d] = v.pos[d];
        }
    }

    double scale = (pos_max - pos_min).maxCoeff();
    for (auto& v : vertices) {
        for (int d = 0; d < dim(); d++) {
			v.pos[d] = (v.pos[d] - pos_min[d]) / scale;
        }
    }
}

void ncMesh::reorderElements(std::vector<int>& elementOrder) const
{
    elementOrder.clear();
    elementOrder.reserve(n_elements);

    int cur_level = 0;
    while (elementOrder.size() < n_elements) {
        for (int order = min_order; order <= max_order; order++) {
            for (int i = 0; i < n_elements; i++) {
                const auto& elem = elements[valid2All(i)];
                if (elem.level != cur_level)
                    continue;
                if (elem.order != order)
                    continue;
                
                elementOrder.push_back(valid2All(i));
            }
        }
        cur_level++;
    }
}

void ncMesh::reorderElements(std::vector<std::vector<int> >& elementOrder) const
{
    int max_level = 0;
    for (const auto& elem : elements) {
        if (elem.is_valid() && max_level < elem.level)
            max_level = elem.level;
    }
    elementOrder.clear();
    elementOrder.resize((max_level + 1) * (max_order - min_order + 1));
    int N = 0;
    int cur_level = 0;
    while (cur_level <= max_level) {
        int order = min_order;
        while (order <= max_order) {
            int cur_bucket = (max_order - min_order + 1) * cur_level + (order - min_order);
            for (int i = 0; i < n_elements; i++) {
                const auto& elem = elements[valid2All(i)];
                if (elem.level != cur_level || elem.order != order)
                    continue;

                N++;
                elementOrder[cur_bucket].push_back(valid2All(i));
            }
            order++;
        }
        cur_level++;
    }
}

void ncMesh::initFromHistory(const json& input)
{
    std::vector<int> tmp_refineHistory;
    for (const auto& i : input["refine"])
        tmp_refineHistory.push_back(i);
    std::vector<int> orders;
    for (const auto& o : input["order"])
        orders.push_back(o);

    for (int id : tmp_refineHistory) {
        if (elements[id].is_valid())
            refineElement(id);
        else
            coarsenElement(elements[id].children(0));
    }

    if (dim() == 3) {
        for (int e = 0; e < elements.size(); e++) {
            const auto& vs = input["element_vertices"][e];
            for (int v = 0; v < vs.size(); v++) {
                bool flag = false;
                for (int w = 0; w < elements[e].vertices.size(); w++)
                    if (elements[e].vertices[w] == vs[v])
                        flag = true;
                assert(flag);
            }
            for (int v = 0; v < vs.size(); v++) {
                elements[e].vertices(v) = vs[v];
            }
        }

        if (input["element_vertices"].size() != elements.size()) {
            logger().error("{} not equal {}!", input["element_vertices"].size(), elements.size());
            exit(0);
        }
    }

    int id = 0;
    max_order = 0;
    min_order = 100;
    for (int o : orders) {
        elements[id].order = o;
        max_order = std::max(o, max_order);
        min_order = std::min(o, min_order);
        id++;
    }
    assert(id == elements.size());
}

void ncMesh::traverseEdge(Eigen::Vector2i v, double p1, double p2, int depth, std::vector<slave_edge>& list) const
{
    int v_mid = findVertex(v);
    std::vector<slave_edge> list1, list2;
    if (v_mid >= 0) {
        double p_mid = (p1 + p2) / 2;
        traverseEdge(Eigen::Vector2i(v[0], v_mid), p1, p_mid, depth+1, list1);
        list.insert(
            list.end(),
            std::make_move_iterator(list1.begin()),
            std::make_move_iterator(list1.end())
        );
        traverseEdge(Eigen::Vector2i(v_mid, v[1]), p_mid, p2, depth+1, list2);
        list.insert(
            list.end(),
            std::make_move_iterator(list2.begin()),
            std::make_move_iterator(list2.end())
        );
    }
    if (depth > 0) {
        int slave_id = findEdge(v);
        if (slave_id >= 0 && edges[slave_id].n_elem() > 0)
            list.emplace_back(slave_id, p1, p2);
    }
}

void ncMesh::compress(Eigen::MatrixXd& v, Eigen::MatrixXi& f) const
{
    v.setConstant(n_verts, dim(), 0);
    for (int i = 0, j = 0; i < vertices.size(); i++) {
        if (vertices[i].n_elem) {
            v.row(j) = vertices[i].pos;
            j++;
        }
    }

    f.setConstant(n_elements, dim() + 1, -1);
    for (int i = 0, j = 0; i < elements.size(); i++) {
        if (elements[i].is_valid()) {
            for (int lv = 0; lv < elements[i].vertices.size(); lv++)
                f(j, lv) = all2ValidVertex(elements[i].geom_vertices(lv));
            j++;
        }
    }
}

void ncMesh::buildPartialMesh(const std::vector<int>& elem_list, Eigen::MatrixXd& v, Eigen::MatrixXi& f) const
{
    std::vector<int> vert_flag(vertices.size(), 0), vert_map(vertices.size(), -1);
    for (const int e : elem_list) {
        const auto& elem = elements[e];
        for (int i = 0; i < elem.vertices.size(); i++) {
            vert_flag[elem.vertices(i)] = 1;
        }
    }

    vert_map[0] = vert_flag[0];
    for (int i = 1; i < vert_flag.size(); i++)
        vert_map[i] = vert_map[i-1] + vert_flag[i];
    const int n_vert = vert_map[vert_map.size() - 1];

    v.setConstant(n_vert, dim(), 0);
    f.setConstant(elem_list.size(), dim() + 1, -1);

    for (int e = 0; e < elem_list.size(); e++) {
        for (int i = 0; i < f.cols(); i++)
            f(e, i) = vert_map[elements[elem_list[e]].vertices(i)] - 1;
    }
    for (int i = 0, j = 0; i < vertices.size(); i++) {
        if (vert_flag[i]) {
            v.row(j) = vertices[i].pos;
            j++;
        }
    }
}

void ncMesh::buildIndexMapping()
{
    all2ValidMap.assign(elements.size(), -1);
    valid2AllMap.resize(n_elements);

    for (int i = 0, e = 0; i < elements.size(); i++) {
        if (elements[i].is_not_valid())
            continue;
        all2ValidMap[i] = e;
        valid2AllMap[e] = i;
        e++;
    }

    n_verts = 0;
    for (int i = 0; i < vertices.size(); i++)
        if (vertices[i].n_elem)
            n_verts++;

    all2ValidVertexMap.assign(vertices.size(), -1);
    valid2AllVertexMap.resize(n_verts);

    for (int i = 0, j = 0; i < vertices.size(); i++) {
        if (vertices[i].n_elem == 0)
            continue;
        all2ValidVertexMap[i] = j;
        valid2AllVertexMap[j] = i;
        j++;
    }
}

}
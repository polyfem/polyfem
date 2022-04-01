#include <polyfem/NCMesh2D.hpp>

#include <polyfem/Logger.hpp>

#include <igl/writeOBJ.h>

#include <polyfem/MeshUtils.hpp>

namespace polyfem
{
    bool NCMesh2D::build_from_matrices(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
    {
        GEO::Mesh mesh_;
		mesh_.clear(false, false);
		to_geogram_mesh(V, F, mesh_);
        orient_normals_2d(mesh_);

        n_elements = 0; 
        vertices.reserve(V.rows());
        for (int i = 0; i < V.rows(); i++) {
            vertices.emplace_back(V.row(i));
        }
        for (int i = 0; i < F.rows(); i++) {
            addElement(F.row(i), -1);
        }

        return true;
    }

    bool NCMesh2D::save(const std::string &path) const
    {
        Eigen::MatrixXd v;
        v.setConstant(n_vertices(), 2, 0);
        for (int i = 0, j = 0; i < vertices.size(); i++) {
            if (vertices[i].n_elem) {
                v.row(j) = vertices[i].pos;
                j++;
            }
        }

        Eigen::MatrixXi f;
        f.setConstant(n_elements, 2 + 1, -1);
        for (int i = 0, j = 0; i < elements.size(); i++) {
            if (elements[i].is_valid()) {
                for (int lv = 0; lv < elements[i].vertices.size(); lv++)
                    f(j, lv) = all2ValidVertex(elements[i].geom_vertices(lv));
                j++;
            }
        }

        igl::writeOBJ(path, v, f);

        return true;
    }


    int NCMesh2D::addElement(Eigen::Vector3i v, int parent)
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

    void NCMesh2D::refineElement(int id)
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

        refineHistory.push_back(id);
    }

    void NCMesh2D::coarsenElement(int id)
    {
        const int parent_id = elements[id].parent;
        auto& parent = elements[parent_id];

        for (int i = 0; i < parent.children.size(); i++)
            assert(elements[parent.children(i)].is_valid() && "Invalid siblings in coarsening!");

        // remove elements
        for (int i = 0; i < parent.children.size(); i++) {
            auto& elem = elements[parent.children(i)];
            elem.is_ghost = true;
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

    int NCMesh2D::globalEdge2LocalEdge(const int e, const int l) const
    {
        for (int i = 0; i < 3; i++) {
            if (elements[e].edges[i] == l)
                return i;
        }
        assert(false);
        return 0;
    }

    int find(const Eigen::VectorXi& vec, int x)
    {
        for (int i = 0; i < 3; i++) {
            if (x == vec[i])
                return i;
        }
        return -1;
    }

    void NCMesh2D::buildEdgeSlaveChain()
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

    void NCMesh2D::markBoundary()
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

    void NCMesh2D::buildElementVertexAdjacency()
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

    Eigen::Vector2d NCMesh2D::edgeWeight2ElemWeight(const int l, const double w)
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

    double NCMesh2D::elemWeight2EdgeWeight(const int l, const Eigen::Vector2d& pos)
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

    void NCMesh2D::buildIndexMapping()
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

        const int n_verts = n_vertices();

        all2ValidVertexMap.assign(vertices.size(), -1);
        valid2AllVertexMap.resize(n_verts);

        for (int i = 0, j = 0; i < vertices.size(); i++) {
            if (vertices[i].n_elem == 0)
                continue;
            all2ValidVertexMap[i] = j;
            valid2AllVertexMap[j] = i;
            j++;
        }

        all2ValidEdgeMap.assign(edges.size(), -1);
        valid2AllEdgeMap.resize(n_edges());

        for (int i = 0, j = 0; i < edges.size(); i++) {
            if (edges[i].n_elem() == 0)
                continue;
            all2ValidEdgeMap[i] = j;
            valid2AllEdgeMap[j] = i;
            j++;
        }
    }

    void NCMesh2D::traverseEdge(Eigen::Vector2i v, double p1, double p2, int depth, std::vector<slave_edge>& list) const
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

    bool NCMesh2D::load(const std::string &path)
    {
        assert(false);
        return false;
    }

    bool NCMesh2D::load(const GEO::Mesh &mesh)
    {
        GEO::Mesh mesh_;
		mesh_.clear(false, false);
		mesh_.copy(mesh);
        orient_normals_2d(mesh_);

        Eigen::MatrixXd V(mesh_.vertices.nb(), 2);
        Eigen::MatrixXi F(mesh_.facets.nb(), 3);

        for (int v = 0; v < V.rows(); v++) {
            const double *ptr = mesh_.vertices.point_ptr(v);
            V.row(v) << ptr[0], ptr[1];
        }

        for (int f = 0; f < F.rows(); f++)
            for (int i = 0; i < F.cols(); i++)
                F(f, i) = mesh_.facets.vertex(f, i);

        n_elements = 0; 
        vertices.reserve(V.rows());
        for (int i = 0; i < V.rows(); i++) {
            vertices.emplace_back(V.row(i));
        }
        for (int i = 0; i < F.rows(); i++) {
            addElement(F.row(i), -1);
        }

        return true;
    }

    void NCMesh2D::bounding_box(RowVectorNd &min, RowVectorNd &max) const
    {
        min = vertices[0].pos;
        max = vertices[0].pos;

        for (const auto& v : vertices)
        {
            if (v.pos[0] > max[0])
                max[0] = v.pos[0];
            if (v.pos[1] > max[1])
                max[1] = v.pos[1];
            if (v.pos[0] < min[0])
                min[0] = v.pos[0];
            if (v.pos[1] < min[1])
                min[1] = v.pos[1];
        }
    }

    Navigation::Index NCMesh2D::get_index_from_face(int f, int lv) const
    {
        const auto& elem = elements[valid2AllElem(f)];
        
        Navigation::Index idx2;
        idx2.face = f;
        idx2.vertex = all2ValidVertex(elem.vertices(lv));
        idx2.edge = all2ValidEdge(elem.edges(lv));
        idx2.face_corner = -1;

        return idx2;
    }

    Navigation::Index NCMesh2D::switch_vertex(Navigation::Index idx) const
    {
        const auto& elem = elements[valid2AllElem(idx.face)];
        const auto& edge = edges[valid2AllEdge(idx.edge)];

        Navigation::Index idx2;
        idx2.face = idx.face;
        idx2.edge = idx.edge;
        
        int v1 = valid2AllVertex(idx.vertex);
        int v2 = -1;
        for (int i = 0; i < edge.vertices.size(); i++)
            if (edge.vertices(i) != v1) 
            {
                v2 = edge.vertices(i);
                break;
            }

        idx2.vertex = all2ValidVertex(v2);
        idx2.face_corner = -1;

        return idx2;
    }

    Navigation::Index NCMesh2D::switch_edge(Navigation::Index idx) const 
    {
        const auto& elem = elements[valid2AllElem(idx.face)];

        Navigation::Index idx2;
        idx2.face = idx.face;
        idx2.vertex = idx.vertex;
        idx2.face_corner = -1;

        for (int i = 0; i < elem.edges.size(); i++)
        {
            const auto& edge = edges[elem.edges(i)];
            const int valid_edge_id = all2ValidEdge(elem.edges(i));
            if (valid_edge_id != idx.edge && find(edge.vertices, idx.vertex) >= 0)
            {
                idx2.edge = valid_edge_id;
                break;
            }
        }

        return idx2;
    }

    Navigation::Index NCMesh2D::switch_face(Navigation::Index idx) const 
    {
        Navigation::Index idx2;
        idx2.edge = idx.edge;
        idx2.vertex = idx.vertex;
        idx2.face_corner = -1;

        const auto& edge = edges[valid2AllEdge(idx.edge)];
        if (edge.n_elem() == 2)
            idx2.face = all2ValidElem(edge.find_opposite_element(valid2AllElem(idx.face)));
        else
            idx2.face = -1;

        return idx2;
    }

    void NCMesh2D::normalize()
    {
        polyfem::RowVectorNd min, max;
        bounding_box(min, max);

        auto extent = max - min;
        double scale = std::max(extent(0), extent(1));

        for (auto &v : vertices)
            v.pos = (v.pos - min) / scale;
    }

    double NCMesh2D::edge_length(const int gid) const
    {
        const int v1 = edge_vertex(gid, 0);
        const int v2 = edge_vertex(gid, 1);
        
        return (point(v1) - point(v2)).norm();
    }

    void NCMesh2D::compute_elements_tag()
    {
        elements_tag_.assign(n_faces(), ElementType::Simplex);
    }
    void NCMesh2D::update_elements_tag()
    {
        elements_tag_.assign(n_faces(), ElementType::Simplex);
    }

    void NCMesh2D::set_point(const int global_index, const RowVectorNd &p)
    {
        vertices[valid2AllVertex(global_index)].pos = p;
    }

    RowVectorNd NCMesh2D::edge_barycenter(const int index) const
    {
        const int v1 = edge_vertex(index, 0);
        const int v2 = edge_vertex(index, 1);

        return 0.5 * (point(v1) + point(v2));
    }

    void NCMesh2D::triangulate_faces(Eigen::MatrixXi &tris, Eigen::MatrixXd &pts, std::vector<int> &ranges) const
    {
		ranges.clear();

		std::vector<Eigen::MatrixXi> local_tris(n_faces());
		std::vector<Eigen::MatrixXd> local_pts(n_faces());

		int total_tris = 0;
		int total_pts = 0;

		ranges.push_back(0);

		for (int f = 0; f < n_faces(); ++f)
		{
			const int n_vertices = n_face_vertices(f);

			Eigen::MatrixXd face_pts(n_vertices, 2);
			local_tris[f].resize(n_vertices - 2, 3);

			for (int i = 0; i < n_vertices; ++i)
			{
				const int vertex = face_vertex(f, i);
				auto pt = point(vertex);
				face_pts(i, 0) = pt[0];
				face_pts(i, 1) = pt[1];
			}

			for (int i = 1; i < n_vertices - 1; ++i)
			{
				local_tris[f].row(i - 1) << 0, i, i + 1;
			}

			local_pts[f] = face_pts;

			total_tris += local_tris[f].rows();
			total_pts += local_pts[f].rows();

			ranges.push_back(total_tris);

			assert(local_pts[f].rows() == face_pts.rows());
		}

		tris.resize(total_tris, 3);
		pts.resize(total_pts, 2);

		int tri_index = 0;
		int pts_index = 0;
		for (std::size_t i = 0; i < local_tris.size(); ++i)
		{
			tris.block(tri_index, 0, local_tris[i].rows(), local_tris[i].cols()) = local_tris[i].array() + pts_index;
			tri_index += local_tris[i].rows();

			pts.block(pts_index, 0, local_pts[i].rows(), local_pts[i].cols()) = local_pts[i];
			pts_index += local_pts[i].rows();
		}
	}

}
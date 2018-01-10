#include "SplineBasis2d.hpp"

#include "QuadraticBSpline2d.hpp"
#include "QuadQuadrature.hpp"
#include "QuadBoundarySampler.hpp"
#include "ElementAssemblyValues.hpp"

#include "FEBasis2d.hpp"


#include <cassert>
#include <iostream>
#include <vector>
#include <array>
#include <map>

#include "UIState.hpp"

namespace poly_fem
{
    using namespace Eigen;

    namespace
    {
        typedef Matrix<std::vector<int>, 3, 3> SpaceMatrix;
        typedef Matrix<std::vector<MatrixXd>, 3, 3> NodeMatrix;


        void print_local_space(const SpaceMatrix &space)
        {
            for(int j=2; j >=0; --j)
            {
                for(int i=0; i < 3; ++i)
                {
                    if(space(i, j).size() > 0){
                        for(std::size_t l = 0; l < space(i, j).size(); ++l)
                            std::cout<<space(i, j)[l]<<",";

                        std::cout<<"\t";
                    }
                    else
                        std::cout<<"x\t";
                }
                std::cout<<std::endl;
            }
        }

        void explore_direction(const Navigation::Index &index, const Mesh2D &mesh, const int x, const int y, const bool is_x, const bool invert, const int b_flag, SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, InterfaceData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
        {
            int node_id;
            bool real_boundary = mesh.node_id_from_edge_index(index, node_id);

            assert(std::find(space(x, y).begin(), space(x, y).end(), node_id) == space(x, y).end());
            space(x, y).push_back(node_id);
            node(x, y).push_back(mesh.node_from_edge_index(index));

            const int x1 =  is_x ? x : (invert? 2 : 0);
            const int y1 = !is_x ? y : (invert? 2 : 0);

            const int x2 =  is_x ? x : (invert? 0 : 2);
            const int y2 = !is_x ? y : (invert? 0 : 2);

            if(node_id < mesh.n_elements())
            {
                Navigation::Index start_index = mesh.switch_face(index);
                assert(start_index.face == node_id);
                assert(start_index.vertex == index.vertex);

                Navigation::Index edge1 = mesh.switch_edge(start_index);
                real_boundary = mesh.node_id_from_edge_index(edge1, node_id);
                if(real_boundary)
                    bounday_nodes.push_back(node_id);
                if(std::find(space(x1, y1).begin(), space(x1, y1).end(), node_id) == space(x1, y1).end())
                {
                    space(x1, y1).push_back(node_id);
                    node(x1, y1).push_back(mesh.node_from_edge_index(edge1));
                }

                Navigation::Index edge2 = mesh.switch_edge(mesh.switch_vertex(start_index));
                real_boundary = mesh.node_id_from_edge_index(edge2, node_id);
                if(real_boundary)
                    bounday_nodes.push_back(node_id);
                if(std::find(space(x2, y2).begin(), space(x2, y2).end(), node_id) == space(x2, y2).end())
                {
                    space(x2, y2).push_back(node_id);
                    node(x2, y2).push_back(mesh.node_from_edge_index(edge2));
                }
            }
            else
            {
                if(real_boundary)
                {
                    switch(b_flag)
                    {
                        case InterfaceData::RIGHT_FLAG: local_boundary.set_right_boundary(); local_boundary.set_right_edge_id(index.edge); break;
                        case InterfaceData::BOTTOM_FLAG: local_boundary.set_bottom_boundary(); local_boundary.set_bottom_edge_id(index.edge); break;
                        case InterfaceData::LEFT_FLAG: local_boundary.set_left_boundary(); local_boundary.set_left_edge_id(index.edge); break;
                        case InterfaceData::TOP_FLAG: local_boundary.set_top_boundary(); local_boundary.set_top_edge_id(index.edge); break;
                    }
                    bounday_nodes.push_back(node_id);
                }
                else
                {
                    InterfaceData &data = poly_edge_to_data[index.edge];
                    // data.element_id = el_index;
                    data.element_id = index.face;
                    data.node_id.push_back(node_id);
                    data.flag = b_flag;
                    data.local_indices.push_back(y * 3 + x);
                    data.vals.push_back(1);
                }
            }
        }

        void add_id_for_poly(const Navigation::Index &index, const int x1, const int y1, const int x2, const int y2, const SpaceMatrix &space, std::map<int, InterfaceData> &poly_edge_to_data)
        {
            auto it = poly_edge_to_data.find(index.edge);
            if(it != poly_edge_to_data.end())
            {
                InterfaceData &data = it->second;

                assert(space(x1, y1).size() == 1);
                data.node_id.push_back(space(x1, y1).front());
                data.local_indices.push_back(y1 * 3 + x1);
                data.vals.push_back(1);

                assert(space(x2, y2).size() == 1);
                data.node_id.push_back(space(x2, y2).front());
                data.local_indices.push_back(y2 * 3 + x2);
                data.vals.push_back(1);
            }
        }

        int build_local_space(const Mesh2D &mesh, const int el_index,  SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, InterfaceData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
        {
            assert(!mesh.is_volume());

            Navigation::Index index;
            // space.setConstant(-1);


            space(1, 1).push_back(el_index);
            node(1, 1).push_back(mesh.node_from_face(el_index));

            //////////////////////////////////////////
            index = mesh.get_index_from_face(el_index);
            explore_direction(index, mesh, 1, 0, false, false, InterfaceData::BOTTOM_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            //////////////////////////////////////////
            index = mesh.next_around_face(index);
            explore_direction(index, mesh, 2, 1, true, false, InterfaceData::RIGHT_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            //////////////////////////////////////////
            index = mesh.next_around_face(index);
            explore_direction(index, mesh, 1, 2, false, true, InterfaceData::TOP_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            //////////////////////////////////////////
            index = mesh.next_around_face(index);
            explore_direction(index, mesh, 0, 1, true, true, InterfaceData::LEFT_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            //////////////////////////////////////////
            if(space(1, 2).front() >= mesh.n_elements() && space(2, 1).front() >= mesh.n_elements())
            {
                Navigation::Index start_index = mesh.get_index_from_face(el_index);
                start_index = mesh.next_around_face(start_index);
                start_index = mesh.next_around_face(start_index);

                const int node_id = mesh.vertex_node_id(start_index.vertex);
                space(2,2).push_back(node_id);
                node(2,2).push_back(mesh.node_from_vertex(start_index.vertex));

                bounday_nodes.push_back(node_id);
            }

            if(space(1, 0).front() >= mesh.n_elements() && space(2, 1).front() >= mesh.n_elements())
            {
                Navigation::Index start_index = mesh.get_index_from_face(el_index);
                start_index = mesh.next_around_face(start_index);

                const int node_id = mesh.vertex_node_id(start_index.vertex);
                space(2,0).push_back(node_id);
                node(2,0).push_back(mesh.node_from_vertex(start_index.vertex));

                bounday_nodes.push_back(node_id);
            }

            if(space(1, 2).front() >= mesh.n_elements() && space(0, 1).front() >= mesh.n_elements())
            {
                Navigation::Index start_index = mesh.get_index_from_face(el_index);
                start_index = mesh.next_around_face(start_index);
                start_index = mesh.next_around_face(start_index);
                start_index = mesh.next_around_face(start_index);

                const int node_id = mesh.vertex_node_id(start_index.vertex);
                space(0,2).push_back(node_id);
                node(0,2).push_back(mesh.node_from_vertex(start_index.vertex));

                bounday_nodes.push_back(node_id);
            }

            if(space(1, 0).front() >= mesh.n_elements() && space(0, 1).front() >= mesh.n_elements())
            {
                Navigation::Index start_index = mesh.get_index_from_face(el_index);

                const int node_id = mesh.vertex_node_id(start_index.vertex);
                space(0,0).push_back(node_id);
                node(0,0).push_back(mesh.node_from_vertex(start_index.vertex));

                bounday_nodes.push_back(node_id);
            }

            // std::cout<<std::endl;
            // print_local_space(space);




            ////////////////////////////////////////////////////////////////////////
            index = mesh.get_index_from_face(el_index);
            add_id_for_poly(index, 0, 0, 2, 0, space, poly_edge_to_data);

            index = mesh.next_around_face(index);
            add_id_for_poly(index, 2, 0, 2, 2, space, poly_edge_to_data);

            index = mesh.next_around_face(index);
            add_id_for_poly(index, 2, 2, 0, 2, space, poly_edge_to_data);

            index = mesh.next_around_face(index);
            add_id_for_poly(index, 0, 2, 0, 0, space, poly_edge_to_data);

            int minCoeff = 0;
            int maxCoeff = -1;

            for(int i = 0; i < 3; ++i)
            {
                for(int j = 0; j < 3; ++j)
                {
                    assert(space(i,j).size() >= 1);
                    for(std::size_t k = 0; k < space(i,j).size(); ++k)
                    {
                        minCoeff = std::min(space(i,j)[k], minCoeff);
                        maxCoeff = std::max(space(i,j)[k], maxCoeff);
                    }
                }
            }

            assert(minCoeff >= 0);
            return maxCoeff;
        }

        void setup_knots_vectors(const int n_els, const SpaceMatrix &space, std::array<std::array<double, 4>, 3> &h_knots, std::array<std::array<double, 4>, 3> &v_knots)
        {
            //left and right neigh are absent
            if(space(0,1).front() >= n_els && space(2,1).front() >= n_els)
            {
                h_knots[0] = {{0, 0, 0, 1}};
                h_knots[1] = {{0, 0, 1, 1}};
                h_knots[2] = {{0, 1, 1, 1}};
            }
            //left neigh is absent
            else if(space(0,1).front() >= n_els)
            {
                h_knots[0] = {{0, 0, 0, 1}};
                h_knots[1] = {{0, 0, 1, 2}};
                h_knots[2] = {{0, 1, 2, 3}};
            }
            //right neigh is absent
            else if(space(2,1).front() >= n_els)
            {
                h_knots[0] = {{-2, -1, 0, 1}};
                h_knots[1] = {{-1, 0, 1, 1}};
                h_knots[2] = {{0, 1, 1, 1}};
            }
            else
            {
                h_knots[0] = {{-2, -1, 0, 1}};
                h_knots[1] = {{-1, 0, 1, 2}};
                h_knots[2] = {{0, 1, 2, 3}};
            }


            //top and bottom neigh are absent
            if(space(1,0).front() >= n_els && space(1,2).front() >= n_els)
            {
                v_knots[0] = {{0, 0, 0, 1}};
                v_knots[1] = {{0, 0, 1, 1}};
                v_knots[2] = {{0, 1, 1, 1}};
            }
            //bottom neigh is absent
            else if(space(1,0).front() >= n_els)
            {
                v_knots[0] = {{0, 0, 0, 1}};
                v_knots[1] = {{0, 0, 1, 2}};
                v_knots[2] = {{0, 1, 2, 3}};
            }
            //top neigh is absent
            else if(space(1,2).front() >= n_els)
            {
                v_knots[0] = {{-2, -1, 0, 1}};
                v_knots[1] = {{-1, 0, 1, 1}};
                v_knots[2] = {{0, 1, 1, 1}};
            }
            else
            {
                v_knots[0] = {{-2, -1, 0, 1}};
                v_knots[1] = {{-1, 0, 1, 2}};
                v_knots[2] = {{0, 1, 2, 3}};
            }
        }

        void basis_for_regular_quad(const SpaceMatrix &space, const NodeMatrix &loc_nodes, const std::array<std::array<double, 4>, 3> &h_knots, const std::array<std::array<double, 4>, 3> &v_knots, ElementBases &b)
        {
            for(int y = 0; y < 3; ++y)
            {
                for(int x = 0; x < 3; ++x)
                {
                    if(space(x,y).size() == 1)
                    {
                        const int global_index = space(x, y).front();
                        const Eigen::MatrixXd &node = loc_nodes(x,y).front();

                        const int local_index = y*3 + x;
                        b.bases[local_index].init(global_index, local_index, node);

                        const QuadraticBSpline2d spline(h_knots[x], v_knots[y]);
                        b.bases[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
                        b.bases[local_index].set_grad( [spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });
                    }
                }
            }
        }

        void basis_for_irregulard_quad(const Mesh2D &mesh, const SpaceMatrix &space, const NodeMatrix &loc_nodes, const std::array<std::array<double, 4>, 3> &h_knots, const std::array<std::array<double, 4>, 3> &v_knots, ElementBases &b)
        {
            for(int y = 0; y < 3; ++y)
            {
                for(int x = 0; x < 3; ++x)
                {
                    if(space(x,y).size() > 1)
                    {
                        const int mpx = 1;
                        const int mpy = y;

                        const int mmx = x;
                        const int mmy = 1;

                        std::vector<int> other_indices;
                        const auto &center = b.bases[1*3 + 1].global().front();

                        const auto &el1 = b.bases[mpy*3 + mpx].global().front();
                        const auto &el2 = b.bases[mmy*3 + mmx].global().front();

                        Navigation::Index index = mesh.get_index_from_face(center.index);
                        while(mesh.next_around_vertex(index).face != el1.index && mesh.next_around_vertex(index).face != el2.index)
                        {
                            index = mesh.next_around_face(index);
                        }

                        index = mesh.next_around_vertex(index);

                        Navigation::Index i1 = mesh.next_around_vertex(index);
                        if(i1.face == space(x,y)[0] || i1.face == space(x,y)[1])
                            index = i1;
                        else
                            index = mesh.next_around_vertex(mesh.switch_vertex(index));

                        const int start = index.face == space(x,y)[0] ? space(x,y)[0] : space(x,y)[1];
                        const int end = start == space(x,y)[0] ? space(x,y)[1] : space(x,y)[0];
                        assert(index.face == space(x,y)[0] || index.face == space(x,y)[1]);

                        while(index.face != end)
                        {
                            other_indices.push_back(index.face);
                            index = mesh.next_around_vertex(index);
                        }
                        other_indices.push_back(end);


                        const int local_index = y*3 + x;
                        auto &base = b.bases[local_index];

                        const int k = int(other_indices.size()) + 3;


                        base.global().resize(k);

                        base.global()[0].index = center.index;
                        base.global()[0].val = (4. - k) / k;
                        base.global()[0].node = center.node;

                        base.global()[1].index = el1.index;
                        base.global()[1].val = (4. - k) / k;
                        base.global()[1].node = el1.node;

                        base.global()[2].index = el2.index;
                        base.global()[2].val = (4. - k) / k;
                        base.global()[2].node = el2.node;


                        for(std::size_t n = 0; n < other_indices.size(); ++n)
                        {
                            base.global()[3+n].index = other_indices[n];
                            base.global()[3+n].val = 4./k;
                            base.global()[3+n].node = mesh.node_from_face(other_indices[n]);
                        }


                        const QuadraticBSpline2d spline(h_knots[x], v_knots[y]);
                        b.bases[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
                        b.bases[local_index].set_grad( [spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });
                    }
                }
            }
        }

        void create_q2_nodes(const Mesh2D &mesh, const std::vector<ElementType> &els_tag, const int el_index, std::set<int> &vertex_id, std::set<int> &edge_id, ElementBases &b, std::vector< int > &bounday_nodes, LocalBoundary &local_boundary, int &n_bases)
        {
            const auto is_q2 = [els_tag](const int face_id){ return els_tag[face_id] == ElementType::MultiSingularInteriorCube || els_tag[face_id] == ElementType::SimpleSingularBoundaryCube; };
            const int n_els = mesh.n_elements();

            b.bases.resize(9);

            Navigation::Index index = mesh.get_index_from_face(el_index);
            for (int j = 0; j < 4; ++j)
            {
                int current_vertex_node_id = -1;
                int current_edge_node_id = -1;
                Eigen::Matrix<double, 1, 2> current_edge_node;
                Eigen::MatrixXd current_vertex_node;

                auto e2l = FEBasis2d::quadr_quad_edge_local_nodes(mesh, index);
                int vertex_basis_id = e2l[0];
                int edge_basis_id = e2l[1];

                const int opposite_face = mesh.switch_face(index).face;

                //if the edge/vertex is boundary the it is a Q2 edge
                bool is_vertex_q2 = true;
                bool is_vertex_boundary = false;

                Navigation::Index vindex=index;

                do
                {
                    if(vindex.face < 0)
                    {
                        is_vertex_boundary = true;
                        break;
                    }
                    if(!is_q2(vindex.face) && mesh.n_element_vertices(vindex.face) == 4)
                    {
                        is_vertex_q2 = false;
                        break;
                    }
                    vindex = mesh.next_around_vertex(vindex);
                }
                while(vindex.edge != index.edge);

                if(is_vertex_q2)
                {
                    vindex=index;
                    do
                    {
                        if(vindex.face < 0)
                        {
                            is_vertex_boundary = true;
                            break;
                        }

                        if(!is_q2(vindex.face) && mesh.n_element_vertices(vindex.face) == 4)
                        {
                            is_vertex_q2 = false;
                            break;
                        }
                        vindex = mesh.next_around_vertex_inv(vindex);
                    }
                    while(vindex.edge != index.edge);
                }

                const bool is_edge_q2 = opposite_face < 0 || mesh.n_element_vertices(opposite_face) > 4 || is_q2(opposite_face);

                if (is_edge_q2)
                {
                    const bool is_new_edge = edge_id.insert(index.edge).second;

                    if(is_new_edge)
                    {
                        current_edge_node_id = ++n_bases;
                        current_edge_node = mesh.edge_mid_point(index.edge);

                        if(opposite_face < 0)
                        {
                            bounday_nodes.push_back(current_edge_node_id);
                            switch(edge_basis_id - 4)
                            {
                                case 1: local_boundary.set_right_boundary(); local_boundary.set_right_edge_id(index.edge); break;
                                case 0: local_boundary.set_bottom_boundary(); local_boundary.set_bottom_edge_id(index.edge); break;
                                case 3: local_boundary.set_left_boundary(); local_boundary.set_left_edge_id(index.edge); break;
                                case 2: local_boundary.set_top_boundary(); local_boundary.set_top_edge_id(index.edge); break;
                            }
                        }
                    }
                }

                if(is_vertex_q2)
                {
                    assert(is_edge_q2);
                    const bool is_new_vertex = vertex_id.insert(index.vertex).second;

                    if(is_new_vertex)
                    {
                        current_vertex_node_id = ++n_bases;
                        mesh.point(index.vertex, current_vertex_node);

                        if(is_vertex_boundary)//mesh.is_vertex_boundary(index.vertex))
                            bounday_nodes.push_back(current_vertex_node_id);
                    }
                }

                //init new Q2 nodes
                if(current_vertex_node_id >= 0)
                    b.bases[vertex_basis_id].init(current_vertex_node_id, vertex_basis_id, current_vertex_node);

                if(current_edge_node_id >= 0)
                    b.bases[edge_basis_id].init(current_edge_node_id, edge_basis_id, current_edge_node);

                //set the basis functions
                b.bases[vertex_basis_id].set_basis([vertex_basis_id](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis2d::quadr_quad_basis_value(vertex_basis_id, uv, val); });
                b.bases[vertex_basis_id].set_grad( [vertex_basis_id](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  FEBasis2d::quadr_quad_basis_grad(vertex_basis_id, uv, val); });

                b.bases[edge_basis_id].set_basis([edge_basis_id](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis2d::quadr_quad_basis_value(edge_basis_id, uv, val); });
                b.bases[edge_basis_id].set_grad( [edge_basis_id](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  FEBasis2d::quadr_quad_basis_grad(edge_basis_id, uv, val); });

                index = mesh.next_around_face(index);
            }

            //central node always present
            constexpr int face_basis_id = 8;
            b.bases[face_basis_id].init(++n_bases, face_basis_id, mesh.node_from_face(el_index));
            b.bases[face_basis_id].set_basis([](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis2d::quadr_quad_basis_value(face_basis_id, uv, val); });
            b.bases[face_basis_id].set_grad( [](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  FEBasis2d::quadr_quad_basis_grad(face_basis_id, uv, val); });
        }

        void insert_into_global(const Local2Global &data, std::vector<Local2Global> &vec)
        {
            //ignore small weights
            if(fabs(data.val) <1e-10 )
                return;

            bool found = false;

            for(std::size_t i = 0; i < vec.size(); ++i)
            {
                if(vec[i].index == data.index)
                {
                    // std::cout<<vec[i].val <<" "<< data.val<<" "<<fabs(vec[i].val - data.val)<<std::endl;
                    assert(fabs(vec[i].val - data.val) < 1e-10);
                    assert((vec[i].node - data.node).norm() < 1e-10);
                    found = true;
                    break;
                }
            }

            if(!found)
                vec.push_back(data);
        }

        void assign_q2_weights(const Mesh2D &mesh, const int el_index, std::vector< ElementBases > &bases)
        {
            Eigen::MatrixXd eval_p;
            Navigation::Index index = mesh.get_index_from_face(el_index);
            ElementBases &b = bases[el_index];

            for (int j = 0; j < 4; ++j)
            {
                const int opposite_face = mesh.switch_face(index).face;

                if(opposite_face < 0 || mesh.n_element_vertices(opposite_face) != 4)
                {
                    index = mesh.next_around_face(index);
                    continue;
                }

                const auto param_p = FEBasis2d::quadr_quad_edge_local_nodes_coordinates(mesh, mesh.switch_face(index));
                const auto indices = FEBasis2d::quadr_quad_edge_local_nodes(mesh, index);
                // std::cout<<param_p<<"\n---------\n"<<std::endl;

                const int i0 = indices[0];
                const int i1 = indices[1];
                const int i2 = indices[2];

                const auto &other_bases = bases[opposite_face];

                for(std::size_t i = 0; i < other_bases.bases.size(); ++i)
                {
                    const auto &other_b = other_bases.bases[i];

                    if(other_b.global().empty()) continue;

                    other_b.basis(param_p, eval_p);
                    assert(eval_p.size() == 3);

                    //basis i of element opposite face is zero on this elements
                    if(eval_p.cwiseAbs().maxCoeff() <= 1e-10)
                        continue;

                    for(std::size_t k = 0; k < other_b.global().size(); ++k)
                    {
                        auto glob0 = other_b.global()[k]; glob0.val *= eval_p(0);
                        auto glob1 = other_b.global()[k]; glob1.val *= eval_p(1);
                        auto glob2 = other_b.global()[k]; glob2.val *= eval_p(2);

                        insert_into_global(glob0, b.bases[i0].global());
                        insert_into_global(glob1, b.bases[i1].global());
                        insert_into_global(glob2, b.bases[i2].global());
                    }
                }

                index = mesh.next_around_face(index);
            }
        }

        void setup_data_for_polygons(const Mesh2D &mesh, const int el_index, const ElementBases &b, std::map<int, InterfaceData> &poly_edge_to_data)
        {
            Navigation::Index index = mesh.get_index_from_face(el_index);
            for (int j = 0; j < 4; ++j)
            {
                const int opposite_face = mesh.switch_face(index).face;
                const bool is_neigh_poly = (opposite_face >= 0 && mesh.n_element_vertices(opposite_face) > 4);

                if(is_neigh_poly)
                {
                    int b_flag;

                    auto e2l = FEBasis2d::quadr_quad_edge_local_nodes(mesh, index);
                    int vertex_basis_id = e2l[0];
                    int edge_basis_id = e2l[1];
                    int vertex_basis_id2 = e2l[2];

                    if(edge_basis_id-4 == 0)
                        b_flag = InterfaceData::BOTTOM_FLAG;
                    else if(edge_basis_id-4 == 1)
                        b_flag = InterfaceData::RIGHT_FLAG;
                    else if(edge_basis_id-4 == 2)
                        b_flag = InterfaceData::TOP_FLAG;
                    else
                        b_flag = InterfaceData::LEFT_FLAG;

                    InterfaceData &data = poly_edge_to_data[index.edge];
                    data.element_id = index.face;
                    data.flag = b_flag;

                    const auto &bases_e = b.bases[edge_basis_id];
                    for(std::size_t i = 0; i < bases_e.global().size(); ++i)
                    {
                        data.node_id.push_back(bases_e.global()[i].index);
                        data.local_indices.push_back(edge_basis_id);
                        data.vals.push_back(bases_e.global()[i].val);
                    }

                    const auto &bases_v1 = b.bases[vertex_basis_id];
                    for(std::size_t i = 0; i < bases_v1.global().size(); ++i)
                    {
                        data.node_id.push_back(bases_v1.global()[i].index);
                        data.local_indices.push_back(vertex_basis_id);
                        data.vals.push_back(bases_v1.global()[i].val);
                    }

                    const auto &bases_v2 = b.bases[vertex_basis_id2];
                    for(std::size_t i = 0; i < bases_v2.global().size(); ++i)
                    {
                        data.node_id.push_back(bases_v2.global()[i].index);
                        data.local_indices.push_back(vertex_basis_id2);
                        data.vals.push_back(bases_v2.global()[i].val);
                    }
                }

                index = mesh.next_around_face(index);
            }
        }
    }


    int SplineBasis2d::build_bases(const Mesh2D &mesh, const std::vector<ElementType> &els_tag, const int quadrature_order, std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes, std::map<int, InterfaceData> &poly_edge_to_data)
    {
        using std::max;
        assert(!mesh.is_volume());

        const int n_els = mesh.n_elements();
        bases.resize(n_els);
        local_boundary.resize(n_els);

        bounday_nodes.clear();

        int n_bases = n_els;

        QuadQuadrature quad_quadrature;

        for(int e = 0; e < n_els; ++e)
        {
            if(els_tag[e] != ElementType::RegularInteriorCube && els_tag[e] != ElementType::RegularBoundaryCube && els_tag[e] != ElementType::SimpleSingularInteriorCube)
                continue;

            SpaceMatrix space;
            NodeMatrix loc_nodes;

            const int max_local_base = build_local_space(mesh, e, space, loc_nodes, local_boundary[e], poly_edge_to_data, bounday_nodes);
            n_bases = max(n_bases, max_local_base);

            ElementBases &b=bases[e];
            quad_quadrature.get_quadrature(quadrature_order, b.quadrature);
            b.bases.resize(9);

            std::array<std::array<double, 4>, 3> h_knots;
            std::array<std::array<double, 4>, 3> v_knots;

            setup_knots_vectors(n_els, space, h_knots, v_knots);

            // print_local_space(space);

            basis_for_regular_quad(space, loc_nodes, h_knots, v_knots, b);
            basis_for_irregulard_quad(mesh, space, loc_nodes, h_knots, v_knots, b);
        }

        std::set<int> edge_id;
        std::set<int> vertex_id;

        for(int e = 0; e < n_els; ++e)
        {
            if(els_tag[e] != ElementType::MultiSingularInteriorCube && els_tag[e] != ElementType::SimpleSingularBoundaryCube)
                continue;

            ElementBases &b=bases[e];
            quad_quadrature.get_quadrature(quadrature_order, b.quadrature);

            create_q2_nodes(mesh, els_tag, e, vertex_id, edge_id, b, bounday_nodes, local_boundary[e], n_bases);
        }


        for(int e = 0; e < n_els; ++e)
        {
            if(els_tag[e] != ElementType::MultiSingularInteriorCube && els_tag[e] != ElementType::SimpleSingularBoundaryCube)
                continue;
            assign_q2_weights(mesh, e, bases);
        }

        for(int e = 0; e < n_els; ++e)
        {
            if(els_tag[e] != ElementType::MultiSingularInteriorCube && els_tag[e] != ElementType::SimpleSingularBoundaryCube)
                continue;
            assign_q2_weights(mesh, e, bases);
        }

        for(int e = 0; e < n_els; ++e)
        {
            if(els_tag[e] != ElementType::MultiSingularInteriorCube && els_tag[e] != ElementType::SimpleSingularBoundaryCube)
                continue;
            const ElementBases &b=bases[e];
            setup_data_for_polygons(mesh, e, b, poly_edge_to_data);
        }

        return n_bases+1;
    }

}

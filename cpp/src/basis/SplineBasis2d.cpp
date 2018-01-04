#include "SplineBasis2d.hpp"

#include "QuadraticBSpline2d.hpp"
#include "QuadQuadrature.hpp"
#include "PolygonQuadrature.hpp"
#include "QuadBoundarySampler.hpp"

#include "FEBasis2d.hpp"

#include "Harmonic.hpp"
#include "Biharmonic.hpp"


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

        static const int LEFT_FLAG = 1;
        static const int TOP_FLAG = 2;
        static const int RIGHT_FLAG = 4;
        static const int BOTTOM_FLAG = 8;

        struct BoundaryData
        {
            int face_id = -1;
            int flag;
            std::vector<int> node_id;

            std::vector<int> x, y;
            std::vector<int> local_indices;
        };


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

        void explore_direction(const Navigation::Index &index, const Mesh2D &mesh, const int x, const int y, const bool is_x, const bool invert, const int b_flag, SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, BoundaryData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
        {
            int node_id;
            const bool real_boundary = mesh.node_id_from_edge_index(index, node_id);

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
                mesh.node_id_from_edge_index(edge1, node_id);
                if(std::find(space(x1, y1).begin(), space(x1, y1).end(), node_id) == space(x1, y1).end())
                {
                    space(x1, y1).push_back(node_id);
                    node(x1, y1).push_back(mesh.node_from_edge_index(edge1));
                }

                Navigation::Index edge2 = mesh.switch_edge(mesh.switch_vertex(start_index));
                mesh.node_id_from_edge_index(edge2, node_id);
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
                        case RIGHT_FLAG: local_boundary.set_right_boundary(); local_boundary.set_right_edge_id(index.edge); break;
                        case BOTTOM_FLAG: local_boundary.set_bottom_boundary(); local_boundary.set_bottom_edge_id(index.edge); break;
                        case LEFT_FLAG: local_boundary.set_left_boundary(); local_boundary.set_left_edge_id(index.edge); break;
                        case TOP_FLAG: local_boundary.set_top_boundary(); local_boundary.set_top_edge_id(index.edge); break;
                    }
                    bounday_nodes.push_back(node_id);
                }
                else
                {
                    BoundaryData &data = poly_edge_to_data[index.edge];
// data.face_id = el_index;
                    data.face_id = index.face;
                    data.node_id.push_back(node_id);
                    data.flag = b_flag;
                    data.x.push_back(x);
                    data.y.push_back(y);
                }
            }
        }

        void add_id_for_poly(const Navigation::Index &index, const int x1, const int y1, const int x2, const int y2, const SpaceMatrix &space, std::map<int, BoundaryData> &poly_edge_to_data)
        {
            auto it = poly_edge_to_data.find(index.edge);
            if(it != poly_edge_to_data.end())
            {
                BoundaryData &data = it->second;

                assert(space(x1, y1).size() == 1);
                data.node_id.push_back(space(x1, y1).front());
                data.x.push_back(x1);
                data.y.push_back(y1);

                assert(space(x2, y2).size() == 1);
                data.node_id.push_back(space(x2, y2).front());
                data.x.push_back(x2);
                data.y.push_back(y2);
            }
        }

        int build_local_space(const Mesh2D &mesh, const int el_index,  SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, BoundaryData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
        {
            assert(!mesh.is_volume());

            Navigation::Index index;
// space.setConstant(-1);


            space(1, 1).push_back(el_index);
            node(1, 1).push_back(mesh.node_from_face(el_index));

//////////////////////////////////////////
            index = mesh.get_index_from_face(el_index);
            explore_direction(index, mesh, 0, 1, true, true, RIGHT_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

//////////////////////////////////////////
            index = mesh.next_around_face(index);
            explore_direction(index, mesh, 1, 0, false, false, TOP_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

//////////////////////////////////////////
            index = mesh.next_around_face(index);
            explore_direction(index, mesh, 2, 1, true, false, LEFT_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

//////////////////////////////////////////
            index = mesh.next_around_face(index);
            explore_direction(index, mesh, 1, 2, false, true, BOTTOM_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

//////////////////////////////////////////
            if(space(1, 0).front() >= mesh.n_elements() && space(0, 1).front() >= mesh.n_elements())
            {
                Navigation::Index start_index = mesh.get_index_from_face(el_index);
                start_index = mesh.switch_vertex(start_index);

                const int node_id = mesh.vertex_node_id(start_index.vertex);
                space(0,0).push_back(node_id);
                node(0,0).push_back(mesh.node_from_vertex(start_index.vertex));

                bounday_nodes.push_back(node_id);
            }

            if(space(1, 2).front() >= mesh.n_elements() && space(0, 1).front() >= mesh.n_elements())
            {
                Navigation::Index start_index = mesh.get_index_from_face(el_index);

                const int node_id = mesh.vertex_node_id(start_index.vertex);
                space(0,2).push_back(node_id);
                node(0,2).push_back(mesh.node_from_vertex(start_index.vertex));

                bounday_nodes.push_back(node_id);
            }

            if(space(1, 0).front() >= mesh.n_elements() && space(2, 1).front() >= mesh.n_elements())
            {
                Navigation::Index start_index = mesh.get_index_from_face(el_index);
                start_index = mesh.switch_vertex(mesh.next_around_face(start_index));

                const int node_id = mesh.vertex_node_id(start_index.vertex);
                space(2,0).push_back(node_id);
                node(2,0).push_back(mesh.node_from_vertex(start_index.vertex));

                bounday_nodes.push_back(node_id);
            }

            if(space(1, 2).front() >= mesh.n_elements() && space(2, 1).front() >= mesh.n_elements())
            {
                Navigation::Index start_index = mesh.get_index_from_face(el_index);
                start_index = mesh.switch_vertex(mesh.switch_edge(start_index));

                const int node_id = mesh.vertex_node_id(start_index.vertex);
                space(2,2).push_back(node_id);
                node(2,2).push_back(mesh.node_from_vertex(start_index.vertex));

                bounday_nodes.push_back(node_id);
            }

// std::cout<<std::endl;
// print_local_space(space);




////////////////////////////////////////////////////////////////////////
            index = mesh.get_index_from_face(el_index);
            add_id_for_poly(index, 0, 0, 0, 2, space, poly_edge_to_data);

            index = mesh.next_around_face(index);
            add_id_for_poly(index, 0, 0, 2, 0, space, poly_edge_to_data);

            index = mesh.next_around_face(index);
            add_id_for_poly(index, 2, 0, 2, 2, space, poly_edge_to_data);


            index = mesh.next_around_face(index);
            add_id_for_poly(index, 0, 2, 2, 2, space, poly_edge_to_data);

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

        void setup_knots_vectors(const int n_els, const SpaceMatrix &space, std::array<std::vector<double>, 3> &h_knots, std::array<std::vector<double>, 3> &v_knots)
        {
//left and right neigh are absent
            if(space(0,1).front() >= n_els && space(2,1).front() >= n_els)
            {
                h_knots[0] = {0, 0, 0, 1};
                h_knots[1] = {0, 0, 1, 1};
                h_knots[2] = {0, 1, 1, 1};
            }
//left neigh is absent
            else if(space(0,1).front() >= n_els)
            {
                h_knots[0] = {0, 0, 0, 1};
                h_knots[1] = {0, 0, 1, 2};
                h_knots[2] = {0, 1, 2, 3};
            }
//right neigh is absent
            else if(space(2,1).front() >= n_els)
            {
                h_knots[0] = {-2, -1, 0, 1};
                h_knots[1] = {-1, 0, 1, 1};
                h_knots[2] = {0, 1, 1, 1};
            }
            else
            {
                h_knots[0] = {-2, -1, 0, 1};
                h_knots[1] = {-1, 0, 1, 2};
                h_knots[2] = {0, 1, 2, 3};
            }


//top and bottom neigh are absent
            if(space(1,0).front() >= n_els && space(1,2).front() >= n_els)
            {
                v_knots[0] = {0, 0, 0, 1};
                v_knots[1] = {0, 0, 1, 1};
                v_knots[2] = {0, 1, 1, 1};
            }
//bottom neigh is absent
            else if(space(1,0).front() >= n_els)
            {
                v_knots[0] = {0, 0, 0, 1};
                v_knots[1] = {0, 0, 1, 2};
                v_knots[2] = {0, 1, 2, 3};
            }
//top neigh is absent
            else if(space(1,2).front() >= n_els)
            {
                v_knots[0] = {-2, -1, 0, 1};
                v_knots[1] = {-1, 0, 1, 1};
                v_knots[2] = {0, 1, 1, 1};
            }
            else
            {
                v_knots[0] = {-2, -1, 0, 1};
                v_knots[1] = {-1, 0, 1, 2};
                v_knots[2] = {0, 1, 2, 3};
            }
        }

        void basis_for_regular_quad(const SpaceMatrix &space, const NodeMatrix &loc_nodes, const std::array<std::vector<double>, 3> &h_knots, const std::array<std::vector<double>, 3> &v_knots, ElementBases &b)
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

        void basis_for_irregulard_quad(const Mesh2D &mesh, const SpaceMatrix &space, const NodeMatrix &loc_nodes, const std::array<std::vector<double>, 3> &h_knots, const std::array<std::vector<double>, 3> &v_knots, ElementBases &b)
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

        void basis_for_q2(const Mesh2D &mesh, const std::vector<ElementType> &els_tag, const int el_index, std::map<int, int > &vertex_id, std::map<int, int > &edge_id, const SpaceMatrix &space, const NodeMatrix &loc_nodes, ElementBases &b, std::map<int, BoundaryData> &poly_edge_to_data, std::vector< int > &bounday_nodes, int &n_bases)
        {
            static const auto is_q2 = [els_tag](const int face_id){ return els_tag[face_id] == ElementType::MultiSingularInteriorCube || els_tag[face_id] == ElementType::SingularBoundaryCube; };
            const int n_els = mesh.n_elements();

            b.bases.resize(9);

            Navigation::Index index = mesh.get_index_from_face(el_index);
            for (int j = 0; j < 4; ++j)
            {
                int current_vertex_node_id = -1;
                int current_edge_node_id = -1;
                int b_flag;
                Eigen::Matrix<double, 1, 2> current_edge_node;
                Eigen::MatrixXd current_vertex_node;


                int right_node_id, bottom_right_node_id, bottom_node_id, bottom_left_node_id, left_node_id;
                Eigen::MatrixXd right_node, bottom_right_node, bottom_node, bottom_left_node, left_node;

                if(j == 1)
                {
                    right_node_id          = space(0, 1).front();
                    bottom_right_node_id   = space(0, 0).front();
                    bottom_node_id         = space(1, 0).front();
                    bottom_left_node_id    = space(2, 0).front();
                    left_node_id           = space(2, 1).front();

                    right_node          = loc_nodes(0, 1).front();
                    bottom_right_node   = loc_nodes(0, 0).front();
                    bottom_node         = loc_nodes(1, 0).front();
                    bottom_left_node    = loc_nodes(2, 0).front();
                    left_node           = loc_nodes(2, 1).front();

                    b_flag = TOP_FLAG;
                }
                else if( j == 2)
                {
                    right_node_id          = space(1, 0).front();
                    bottom_right_node_id   = space(2, 0).front();
                    bottom_node_id         = space(2, 1).front();
                    bottom_left_node_id    = space(2, 2).front();
                    left_node_id           = space(1, 2).front();

                    right_node          = loc_nodes(1, 0).front();
                    bottom_right_node   = loc_nodes(2, 0).front();
                    bottom_node         = loc_nodes(2, 1).front();
                    bottom_left_node    = loc_nodes(2, 2).front();
                    left_node           = loc_nodes(1, 2).front();

                    b_flag = LEFT_FLAG;
                }
                else if( j == 3)
                {
                    right_node_id          = space(2, 1).front();
                    bottom_right_node_id   = space(2, 2).front();
                    bottom_node_id         = space(1, 2).front();
                    bottom_left_node_id    = space(0, 2).front();
                    left_node_id           = space(0, 1).front();

                    right_node          = loc_nodes(2, 1).front();
                    bottom_right_node   = loc_nodes(2, 2).front();
                    bottom_node         = loc_nodes(1, 2).front();
                    bottom_left_node    = loc_nodes(0, 2).front();
                    left_node           = loc_nodes(0, 1).front();

                    b_flag = BOTTOM_FLAG;
                }
                else
                {
                    right_node_id          = space(1, 2).front();
                    bottom_right_node_id   = space(0, 2).front();
                    bottom_node_id         = space(0, 1).front();
                    bottom_left_node_id    = space(0, 0).front();
                    left_node_id           = space(1, 0).front();

                    right_node          = loc_nodes(1, 2).front();
                    bottom_right_node   = loc_nodes(0, 2).front();
                    bottom_node         = loc_nodes(0, 1).front();
                    bottom_left_node    = loc_nodes(0, 0).front();
                    left_node           = loc_nodes(1, 0).front();

                    b_flag = RIGHT_FLAG;
                }

                const int opposite_face = mesh.switch_face(index).face;
                const int other_face = mesh.switch_face(mesh.switch_edge(index)).face;
                const bool is_vertex_q2 = other_face < 0 || mesh.n_element_vertices(other_face) > 4 || is_q2(other_face);

                if (opposite_face < 0 || mesh.n_element_vertices(opposite_face) > 4) {
                    current_edge_node_id = mesh.edge_node_id(index.edge);
                    current_edge_node = mesh.node_from_edge_index(index);
                    if(opposite_face < 0)
                        bounday_nodes.push_back(current_edge_node_id);

                    if(is_vertex_q2)
                    {
                        auto it = vertex_id.find(index.vertex);

                        if(it == vertex_id.end())
                        {
                            current_vertex_node_id = ++n_bases;
                            vertex_id[index.vertex] = current_vertex_node_id;
                        }
                        else
                            current_vertex_node_id = it->second;

                        if(opposite_face < 0)
                            bounday_nodes.push_back(current_vertex_node_id);
                        mesh.point(index.vertex, current_vertex_node);
                    }
                }
                else
                {
                    const bool is_edge_q2 = is_q2(opposite_face);

                    if(is_edge_q2)
                    {
                        if(is_vertex_q2)
                        {
                            auto it = vertex_id.find(index.vertex);

                            if(it == vertex_id.end())
                            {
                                current_vertex_node_id = ++n_bases;
                                vertex_id[index.vertex] = current_vertex_node_id;
                            }
                            else
                                current_vertex_node_id = it->second;

                            if(other_face < 0)
                                bounday_nodes.push_back(current_vertex_node_id);
                            mesh.point(index.vertex, current_vertex_node);
                        }

                        auto it = edge_id.find(index.edge);

                        if(it == edge_id.end())
                        {
                            current_edge_node_id = ++n_bases;
                            edge_id[index.edge] = current_edge_node_id;
                        }
                        else
                            current_edge_node_id = it->second;

                        current_edge_node = mesh.edge_mid_point(index.edge);
                    }
                }

                if(current_vertex_node_id >= 0)
                {
                    b.bases[2*j].init(current_vertex_node_id, 2*j+1, current_vertex_node);

                    if(opposite_face >= 0 && mesh.n_element_vertices(opposite_face) > 4)
                    {
                        BoundaryData &data = poly_edge_to_data[index.edge];
                        if(data.x.empty())
                        {
                            data.face_id = index.face;
                            data.node_id.push_back(current_vertex_node_id);
                            data.flag = b_flag;
                            data.local_indices.push_back(2*j);
                        }
                        else
                            std::cout<<data.face_id<<std::endl;
                    }

                }
                else
                {
                    // std::cout<<j<<" n "<<right_node_id<< " "<<bottom_right_node_id<< " "<<bottom_node_id<<" "<<bottom_left_node_id<< " "<<left_node_id<<std::endl;

                    auto &global = b.bases[2*j].global();

                    //central
                    if(bottom_right_node_id < n_els)
                        global.push_back(Local2Global(el_index, mesh.node_from_face(el_index), 1./4.));


                    if(bottom_right_node_id < n_els)
                        global.push_back(Local2Global(right_node_id, right_node, 1./4.));
                    else if(right_node_id >= n_els)
                        global.push_back(Local2Global(right_node_id, right_node, 1./2.));


                    if(bottom_right_node_id >= n_els)
                        global.push_back(Local2Global(bottom_right_node_id, bottom_right_node, 1./2.));
                    else
                        global.push_back(Local2Global(bottom_right_node_id, bottom_right_node, 1./4.));


                    if(bottom_right_node_id < n_els)
                        global.push_back(Local2Global(bottom_node_id, bottom_node, 1./4.));
                    else if(bottom_node_id >= n_els)
                        global.push_back(Local2Global(bottom_node_id, bottom_node, 1./2.));
                }



                if(current_edge_node_id >= 0)
                {
                    b.bases[2*j+1].init(current_edge_node_id, 2*j+1, current_edge_node);

                    if(opposite_face >= 0 && mesh.n_element_vertices(opposite_face) > 4)
                    {
                        BoundaryData &data = poly_edge_to_data[index.edge];
                        if(data.x.empty())
                        {
                            data.face_id = index.face;
                            data.node_id.push_back(current_edge_node_id);
                            data.flag = b_flag;
                            data.local_indices.push_back(2*j+1);
                        }
                    }
                }
                else
                {
                    // std::cout<<j<<" e "<<right_node_id<< " "<<bottom_right_node_id<< " "<<bottom_node_id<<" "<<bottom_left_node_id<< " "<<left_node_id<<std::endl;

                    auto &global = b.bases[2*j+1].global();

                    //central
                    if(right_node_id >= n_els || left_node_id >= n_els)
                        global.push_back(Local2Global(el_index, mesh.node_from_face(el_index), 5./16.));
                    else
                        global.push_back(Local2Global(el_index, mesh.node_from_face(el_index), 3./8.));


                    if(right_node_id >= n_els)
                        global.push_back(Local2Global(right_node_id, right_node, 1./8.));
                    else
                        global.push_back(Local2Global(right_node_id, right_node, 1./16.));


                    if(right_node_id >= n_els)
                        global.push_back(Local2Global(bottom_right_node_id, bottom_right_node, 1./8.));
                    else
                        global.push_back(Local2Global(bottom_right_node_id, bottom_right_node, 1./16.));


                    if(right_node_id >= n_els || left_node_id >= n_els)
                        global.push_back(Local2Global(bottom_node_id, bottom_node, 5./16.));
                    else
                        global.push_back(Local2Global(bottom_node_id, bottom_node, 3./8.));


                    if(left_node_id >= n_els)
                        global.push_back(Local2Global(bottom_left_node_id, bottom_left_node, 1./8.));
                    else
                        global.push_back(Local2Global(bottom_left_node_id, bottom_left_node, 1./16.));


                    if(left_node_id >= n_els)
                        global.push_back(Local2Global(left_node_id, left_node, 1./8.));
                    else
                        global.push_back(Local2Global(left_node_id, left_node, 1./16.));
                }

                const int nj = (j+3)%4;

                b.bases[2*j].set_basis([nj](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis2d::quad_basis_basis(2, 2*nj, uv, val); });
                b.bases[2*j].set_grad( [nj](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  FEBasis2d::quad_basis_grad(2, 2*nj, uv, val); });

                b.bases[2*j+1].set_basis([nj](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis2d::quad_basis_basis(2, 2*nj+1, uv, val); });
                b.bases[2*j+1].set_grad( [nj](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  FEBasis2d::quad_basis_grad(2, 2*nj+1, uv, val); });

                index = mesh.next_around_face(index);
            }

            b.bases[8].init(++n_bases, 8, mesh.node_from_face(el_index));
            b.bases[8].set_basis([](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis2d::quad_basis_basis(2, 8, uv, val); });
            b.bases[8].set_grad( [](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  FEBasis2d::quad_basis_grad(2, 8, uv, val); });
        }

        void sample_polygon(const int element_index, const int samples_res, const Mesh2D &mesh, std::map<int, BoundaryData> &poly_edge_to_data, const std::vector< ElementBases > &bases, std::vector<int> &local_to_global, const double eps, const bool c1_continuous, Eigen::MatrixXd &boundary_samples, Eigen::MatrixXd &poly_samples, Eigen::MatrixXd &rhs)
        {
            const int n_edges = mesh.n_element_vertices(element_index);

            const int poly_local_n = (samples_res - 1)/3;
            const int n_samples      = (samples_res - 1) * n_edges;
            const int n_poly_samples = poly_local_n * n_edges;

            boundary_samples.resize(n_samples, 2);
            boundary_samples.setConstant(0);
            poly_samples.resize(n_poly_samples, 2);

            Eigen::MatrixXd samples, mapped, basis_val, grad_basis_val;
            std::vector<Eigen::MatrixXd> grads;

            Navigation::Index index = mesh.get_index_from_face(element_index);
            for(int i = 0; i < n_edges; ++i)
            {
                const BoundaryData &bdata = poly_edge_to_data[index.edge];
                local_to_global.insert(local_to_global.end(), bdata.node_id.begin(), bdata.node_id.end());

                index = mesh.next_around_face(index);
            }

            std::sort( local_to_global.begin(), local_to_global.end() );
            local_to_global.erase( std::unique( local_to_global.begin(), local_to_global.end() ), local_to_global.end() );
            // assert(int(local_to_global.size()) <= n_edges);

            rhs = Eigen::MatrixXd::Zero(n_samples + (c1_continuous? (2*n_samples): 0), local_to_global.size());

            index = mesh.get_index_from_face(element_index);

            //TODO compute first prev!
            Eigen::MatrixXd prev;

            for(int i = 0; i < n_edges; ++i)
            {
                //no boundary polygons
                assert(mesh.switch_face(index).face >= 0);

                const BoundaryData &bdata = poly_edge_to_data[index.edge];
                const ElementBases &b=bases[bdata.face_id];
                assert(bdata.face_id == mesh.switch_face(index).face);

                const bool has_samples = QuadBoundarySampler::sample(bdata.flag == RIGHT_FLAG, bdata.flag == BOTTOM_FLAG, bdata.flag == LEFT_FLAG, bdata.flag == TOP_FLAG, samples_res, false, samples);
                assert(has_samples);

                b.eval_geom_mapping(samples, mapped);

                if(c1_continuous)
                {
                    b.eval_geom_mapping_grads(samples, grads);
                }

                bool must_reverse = true;
                if(prev.size() > 0)
                {
                    const double dist_first = (mapped.row(0)-prev).norm();

                    if(dist_first < 1e-8)
                    {
                        samples = samples.block(1, 0, samples.rows()-1, samples.cols()).eval();
                        mapped = mapped.block(1, 0, mapped.rows()-1, mapped.cols()).eval();

                        must_reverse = false;
                    }
                    else
                    {
                        // assert((mapped.row(mapped.rows()-1) - prev).norm() < 1e-8);

                        samples = samples.block(0, 0, samples.rows()-1, samples.cols()).eval();
                        mapped = mapped.block(0, 0, mapped.rows()-1, mapped.cols()).eval();

                        mapped = mapped.colwise().reverse().eval();

                        must_reverse = true;
                    }
                }
                else
                {
                    // samples = samples.block(0, 0, samples.rows()-1, samples.cols()).eval();
                    // mapped = mapped.block(0, 0, mapped.rows()-1, mapped.cols()).eval();

                    // mapped = mapped.colwise().reverse().eval();
                    // must_reverse = true;

                    samples = samples.block(1, 0, samples.rows()-1, samples.cols()).eval();
                    mapped = mapped.block(1, 0, mapped.rows()-1, mapped.cols()).eval();

                    must_reverse = false;
                }

                // assert(bdata.node_id.size() == 3);
                for(std::size_t bi = 0; bi < bdata.node_id.size(); ++bi)
                {
                    const int local_index = bdata.local_indices.empty() ? (bdata.y[bi] * 3 + bdata.x[bi]) : bdata.local_indices[bi];
                    // assert(b.bases[local_index].global_index() == bdata.node_id[bi]);
                    const long basis_index = std::distance(local_to_global.begin(), std::find(local_to_global.begin(), local_to_global.end(), bdata.node_id[bi]));

                    b.bases[local_index].basis(samples, basis_val);

                    if(must_reverse)
                        basis_val = basis_val.reverse().eval();
                    rhs.block(i*(samples_res-1), basis_index, basis_val.rows(), 1) = basis_val;

                    if(c1_continuous)
                    {
                        b.bases[local_index].grad(samples, grad_basis_val);

                        if(must_reverse)
                            grad_basis_val = grad_basis_val.colwise().reverse().eval();

                        for(long k = 0; k < grad_basis_val.rows(); ++k)
                        {
                            const Eigen::MatrixXd trans_grad = grad_basis_val.row(k) * grads[k];

                            rhs(n_samples + 2*i*(samples_res-1) + 2*k,     basis_index) = trans_grad(0);
                            rhs(n_samples + 2*i*(samples_res-1) + 2*k + 1, basis_index) = trans_grad(1);
                        }
                    }

                }


                prev = mapped.row(mapped.rows()-1);
                assert(mapped.rows() == (samples_res-1));
                boundary_samples.block(i*(samples_res-1), 0, mapped.rows(), mapped.cols()) = mapped;
                const double offset = double(samples_res-1)/(poly_local_n+1);

                for(int j = 0; j < poly_local_n; ++j)
                {
                    const int poly_index = (j+1)*offset;
                    std::cout<<poly_index<<std::endl;

                    if(eps > 0)
                    {
                        const int im = poly_index - 1;
                        const int ip = poly_index + 1;

                        const Eigen::MatrixXd e0 = (mapped.row(poly_index) - mapped.row(im)).normalized();
                        const Eigen::MatrixXd e1 = (mapped.row(ip) - mapped.row(poly_index)).normalized();

                        const Eigen::Vector2d n0(e0(1), -e0(0));
                        const Eigen::Vector2d n1(e1(1), -e1(0));
                        //TODO discad point if inside
                        const Eigen::Vector2d n = (n0+n1).normalized();

                        poly_samples.row(i*poly_local_n+j) = n.transpose()*eps + mapped.row(poly_index);
                    }
                    else
                        poly_samples.row(i*poly_local_n+j) = mapped.row(poly_index);
                }

                index = mesh.next_around_face(index);
            }
        }
    }


    int SplineBasis2d::build_bases(const Mesh2D &mesh, const std::vector<ElementType> &els_tag, const int quadrature_order, std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes, std::map<int, Eigen::MatrixXd> &polys)
    {
        using std::max;
        assert(!mesh.is_volume());

        const int n_els = mesh.n_elements();
        bases.resize(n_els);
        local_boundary.resize(n_els);

        bounday_nodes.clear();

        int n_bases = n_els;

        QuadQuadrature quad_quadrature;

        std::map<int, BoundaryData> poly_edge_to_data;

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

            std::array<std::vector<double>, 3> h_knots;
            std::array<std::vector<double>, 3> v_knots;

            setup_knots_vectors(n_els, space, h_knots, v_knots);

            // print_local_space(space);

            basis_for_regular_quad(space, loc_nodes, h_knots, v_knots, b);
            basis_for_irregulard_quad(mesh, space, loc_nodes, h_knots, v_knots, b);
        }



        std::map<int, int > edge_id;
        std::map<int, int > vertex_id;

        for(int e = 0; e < n_els; ++e)
        {
            if(els_tag[e] != ElementType::MultiSingularInteriorCube && els_tag[e] != ElementType::SingularBoundaryCube)
                continue;

            SpaceMatrix space;
            NodeMatrix loc_nodes;
            std::map<int, BoundaryData> dummy;
            build_local_space(mesh, e, space, loc_nodes, local_boundary[e], dummy, bounday_nodes);
            ElementBases &b=bases[e];
            quad_quadrature.get_quadrature(quadrature_order, b.quadrature);

            basis_for_q2(mesh, els_tag, e, vertex_id, edge_id, space, loc_nodes, b, poly_edge_to_data, bounday_nodes, n_bases);
            // std::cout<<b<<std::endl;

        }

        const int samples_res = 20;
        const bool use_harmonic = true;
        const bool c1_continuous = !use_harmonic && true;


        PolygonQuadrature poly_quad;
        Eigen::Matrix2d det_mat;
        Eigen::MatrixXd p0, p1;

        for(int e = 0; e < n_els; ++e)
        {
            if(els_tag[e] != ElementType::InteriorPolytope && els_tag[e] != ElementType::BoundaryPolytope)
                continue;

            const int n_edges = mesh.n_element_vertices(e);
            double area = 0;
            for(int i = 0; i < n_edges; ++i)
            {
                const int ip = (i + 1) % n_edges;

                mesh.point(mesh.vertex_global_index(e, i), p0);
                mesh.point(mesh.vertex_global_index(e, ip), p1);
                det_mat.row(0) = p0;
                det_mat.row(1) = p1;

                area += det_mat.determinant();
            }
            area = fabs(area);
            // const double eps = use_harmonic ? (0.08*area) : 0;
            const double eps = 0.08*area;

            std::vector<int> local_to_global;
            Eigen::MatrixXd boundary_samples, poly_samples;
            Eigen::MatrixXd rhs;

            sample_polygon(e, samples_res, mesh, poly_edge_to_data, bases, local_to_global, eps, c1_continuous, boundary_samples, poly_samples, rhs);

            ElementBases &b=bases[e];
            b.has_parameterization = false;
            poly_quad.get_quadrature(boundary_samples, quadrature_order, b.quadrature);

            polys[e] = boundary_samples;

            const int n_poly_bases = int(local_to_global.size());
            b.bases.resize(n_poly_bases);

            if(use_harmonic)
            {
                igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
                viewer.data.add_points(poly_samples, Eigen::Vector3d(0,1,1).transpose());

                viewer.data.add_points(boundary_samples, Eigen::Vector3d(1,0,1).transpose());
                // for(int asd = 0; asd < boundary_samples.rows(); ++asd)
                    // viewer.data.add_label(boundary_samples.row(asd), std::to_string(asd));

                Harmonic harmonic(poly_samples, boundary_samples, rhs);

                for(int i = 0; i < n_poly_bases; ++i)
                {
                    b.bases[i].init(local_to_global[i], i, Eigen::MatrixXd(1,2));
                    b.bases[i].set_basis([harmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { harmonic.basis(i, uv, val); });
                    b.bases[i].set_grad( [harmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { harmonic.grad(i, uv, val); });
                }
            }
            else
            {
                Biharmonic biharmonic(poly_samples, boundary_samples, rhs);

                for(int i = 0; i < n_poly_bases; ++i)
                {
                    b.bases[i].init(local_to_global[i], i, Eigen::MatrixXd(1,2));
                    b.bases[i].set_basis([biharmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { biharmonic.basis(i, uv, val); });
                    b.bases[i].set_grad( [biharmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { biharmonic.grad(i, uv, val); });
                }
            }
        }

        return n_bases+1;
    }

}

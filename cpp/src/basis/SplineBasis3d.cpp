#include "SplineBasis3d.hpp"

#include "QuadraticBSpline3d.hpp"
#include "HexQuadrature.hpp"
// #include "PolygonQuadrature.hpp"
// #include "HexBoundarySampler.hpp"

// #include "Harmonic.hpp"
// #include "Biharmonic.hpp"


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
        typedef Matrix<std::vector<int>, 3, 3> Space2d;
        typedef Matrix<std::vector<MatrixXd>, 3, 3> Node2d;

        class SpaceMatrix
        {
        public:
            inline const std::vector<int> &operator()(const int i, const int j, const int k) const
            {
                return space_[k](i,j);
            }

            inline std::vector<int> &operator()(const int i, const int j, const int k)
            {
                return space_[k](i,j);
            }
        private:
            std::array<Space2d, 3> space_;
        };

        class NodeMatrix
        {
        public:
            inline const std::vector<MatrixXd> &operator()(const int i, const int j, const int k) const
            {
                return node_[k](i,j);
            }

            inline std::vector<MatrixXd> &operator()(const int i, const int j, const int k)
            {
                return node_[k](i,j);
            }
        private:
            std::array<Node2d, 3> node_;
        };


        static const int LEFT_FLAG = 1;
        static const int TOP_FLAG = 2;
        static const int RIGHT_FLAG = 4;
        static const int BOTTOM_FLAG = 8;
        static const int FRONT_FLAG = 16;
        static const int BACK_FLAG = 32;


        void print_local_space(const SpaceMatrix &space)
        {
            for(int k = 2; k >= 0; --k)
            {
                for(int j=2; j >=0; --j)
                {
                    for(int i=0; i < 3; ++i)
                    {
                        if(space(i, j, k).size() > 0){
                            for(std::size_t l = 0; l < space(i, j, k).size(); ++l)
                                std::cout<<space(i, j, k)[l]<<",";

                            std::cout<<"\t";
                        }
                        else
                            std::cout<<"x\t";
                    }
                    std::cout<<std::endl;
                }

                std::cout<<"\n"<<std::endl;
            }
        }

        void explore_edge(const Navigation3D::Index &index, const Mesh3D &mesh, const int x, const int y, const int z, SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, InterfaceData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
        {
            int node_id;
            bool boundary = mesh.node_id_from_edge_index(index, node_id);

            assert(std::find(space(x, y, z).begin(), space(x, y, z).end(), node_id) == space(x, y, z).end());
            space(x, y, z).push_back(node_id);
            node(x, y, z).push_back(mesh.node_from_edge_index(index));

            if(node_id >= mesh.n_elements() && boundary)
                bounday_nodes.push_back(node_id);
        }

        void explore_vertex(const Navigation3D::Index &index, const Mesh3D &mesh, const int x, const int y, const int z, SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, InterfaceData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
        {
            int node_id;
            bool boundary = mesh.node_id_from_vertex_index(index, node_id);

            assert(std::find(space(x, y, z).begin(), space(x, y, z).end(), node_id) == space(x, y, z).end());
            space(x, y, z).push_back(node_id);
            node(x, y, z).push_back(mesh.node_from_vertex_index(index));

            if(node_id >= mesh.n_elements() && boundary)
                bounday_nodes.push_back(node_id);
        }

        void explore_face(const Navigation3D::Index &index, const Mesh3D &mesh, const int x, const int y, const int z,  const int b_flag, SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, InterfaceData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
        {
            int node_id;
            const bool real_boundary = mesh.node_id_from_face_index(index, node_id);

            assert(std::find(space(x, y, z).begin(), space(x, y, z).end(), node_id) == space(x, y, z).end());
            space(x, y, z).push_back(node_id);
            node(x, y, z).push_back(mesh.node_from_face_index(index));




            if(node_id >= mesh.n_elements())
            {
                if(real_boundary)
                {
                    switch(b_flag)
                    {
                        case RIGHT_FLAG: local_boundary.set_right_boundary(); local_boundary.set_right_edge_id(index.face); break;
                        case BOTTOM_FLAG: local_boundary.set_bottom_boundary(); local_boundary.set_bottom_edge_id(index.face); break;
                        case LEFT_FLAG: local_boundary.set_left_boundary(); local_boundary.set_left_edge_id(index.face); break;
                        case TOP_FLAG: local_boundary.set_top_boundary(); local_boundary.set_top_edge_id(index.face); break;
                        case FRONT_FLAG: local_boundary.set_front_boundary(); local_boundary.set_front_edge_id(index.face); break;
                        case BACK_FLAG: local_boundary.set_back_boundary(); local_boundary.set_back_edge_id(index.face); break;
                    }
                    bounday_nodes.push_back(node_id);
                }
                else
                {
            //         BoundaryData &data = poly_edge_to_data[index.edge];
            //         // data.face_id = el_index;
            //         data.face_id = index.face;
            //         data.node_id.push_back(node_id);
            //         data.flag = b_flag;
            //         data.x.push_back(x);
            //         data.y.push_back(y);
                }
            }
        }

        int build_local_space(const Mesh3D &mesh, const int el_index,  SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, InterfaceData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
        {
            assert(mesh.is_volume());

            Navigation3D::Index start_index = mesh.get_index_from_element(el_index);
            Navigation3D::Index index;

            std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> to_face;
            mesh.to_face_functions(to_face);

            std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 12> to_edge;
            mesh.to_edge_functions(to_edge);

            std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 8> to_vertex;
            mesh.to_vertex_functions(to_vertex);

            space(1, 1, 1).push_back(el_index);
            node(1, 1, 1).push_back(mesh.node_from_element(el_index));

            ///////////////////////
            index = to_face[1](start_index);
            explore_face(index, mesh, 1, 1, 2, TOP_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_face[0](start_index);
            explore_face(index, mesh, 1, 1, 0, BOTTOM_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_face[3](start_index);
            explore_face(index, mesh, 0, 1, 1, LEFT_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_face[2](start_index);
            explore_face(index, mesh, 2, 1, 1, RIGHT_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_face[4](start_index);
            explore_face(index, mesh, 1, 0, 1, FRONT_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_face[5](start_index);
            explore_face(index, mesh, 1, 2, 1, BACK_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);


            ///////////////////////
            index = to_edge[0](start_index);
            explore_edge(index, mesh, 1, 2, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[1](start_index);
            explore_edge(index, mesh, 2, 1, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[2](start_index);
            explore_edge(index, mesh, 1, 0, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[3](start_index);
            explore_edge(index, mesh, 0, 1, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);


            index = to_edge[4](start_index);
            explore_edge(index, mesh, 0, 2, 1, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[5](start_index);
            explore_edge(index, mesh, 2, 2, 1, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[6](start_index);
            explore_edge(index, mesh, 2, 0, 1, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[7](start_index);
            explore_edge(index, mesh, 0, 0, 1, space, node, local_boundary, poly_edge_to_data, bounday_nodes);



            index = to_edge[8](start_index);
            explore_edge(index, mesh, 1, 2, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[9](start_index);
            explore_edge(index, mesh, 2, 1, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[10](start_index);
            explore_edge(index, mesh, 1, 0, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[11](start_index);
            explore_edge(index, mesh, 0, 1, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);


            ////////////////////////////////////////////////////////////////////////
            index = to_vertex[0](start_index);
            explore_vertex(index, mesh, 0, 2, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_vertex[1](start_index);
            explore_vertex(index, mesh, 2, 2, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_vertex[2](start_index);
            explore_vertex(index, mesh, 2, 0, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_vertex[3](start_index);
            explore_vertex(index, mesh, 0, 0, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);



            index = to_vertex[4](start_index);
            explore_vertex(index, mesh, 0, 2, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_vertex[5](start_index);
            explore_vertex(index, mesh, 2, 2, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_vertex[6](start_index);
            explore_vertex(index, mesh, 2, 0, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_vertex[7](start_index);
            explore_vertex(index, mesh, 0, 0, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            // std::cout<<std::endl;
            // print_local_space(space);




            // ////////////////////////////////////////////////////////////////////////
            // index = mesh.get_index_from_face(el_index);
            // add_id_for_poly(index, 0, 0, 0, 2, space, poly_edge_to_data);

            // index = mesh.next_around_face(index);
            // add_id_for_poly(index, 0, 0, 2, 0, space, poly_edge_to_data);

            // index = mesh.next_around_face(index);
            // add_id_for_poly(index, 2, 0, 2, 2, space, poly_edge_to_data);


            // index = mesh.next_around_face(index);
            // add_id_for_poly(index, 0, 2, 2, 2, space, poly_edge_to_data);

            int minCoeff = 0;
            int maxCoeff = -1;

            for(int l = 0; l < 3; ++l)
            {
                for(int i = 0; i < 3; ++i)
                {
                    for(int j = 0; j < 3; ++j)
                    {
                        assert(space(i,j,l).size() >= 1);
                        for(std::size_t k = 0; k < space(i,j,l).size(); ++k)
                        {
                            minCoeff = std::min(space(i,j,l)[k], minCoeff);
                            maxCoeff = std::max(space(i,j,l)[k], maxCoeff);
                        }
                    }
                }
            }

            assert(minCoeff >= 0);
            return maxCoeff;
        }

        void setup_knots_vectors(const int n_els, const SpaceMatrix &space, std::array<std::vector<double>, 3> &h_knots, std::array<std::vector<double>, 3> &v_knots, std::array<std::vector<double>, 3> &w_knots)
        {
            //left and right neigh are absent
            if(space(0, 1, 1).front() >= n_els && space(2, 1, 1).front() >= n_els)
            {
                h_knots[0] = {0, 0, 0, 1};
                h_knots[1] = {0, 0, 1, 1};
                h_knots[2] = {0, 1, 1, 1};
            }
             //left neigh is absent
            else if(space(0, 1, 1).front() >= n_els)
            {
                h_knots[0] = {0, 0, 0, 1};
                h_knots[1] = {0, 0, 1, 2};
                h_knots[2] = {0, 1, 2, 3};
            }
            //right neigh is absent
            else if(space(2,1,1).front() >= n_els)
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
            if(space(1,0,1).front() >= n_els && space(1,2,1).front() >= n_els)
            {
                v_knots[0] = {0, 0, 0, 1};
                v_knots[1] = {0, 0, 1, 1};
                v_knots[2] = {0, 1, 1, 1};
            }
            //bottom neigh is absent
            else if(space(1,0,1).front() >= n_els)
            {
                v_knots[0] = {0, 0, 0, 1};
                v_knots[1] = {0, 0, 1, 2};
                v_knots[2] = {0, 1, 2, 3};
            }
            //top neigh is absent
            else if(space(1,2,1).front() >= n_els)
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


            //front and back neigh are absent
            if(space(1,1,0).front() >= n_els && space(1,1,2).front() >= n_els)
            {
                w_knots[0] = {0, 0, 0, 1};
                w_knots[1] = {0, 0, 1, 1};
                w_knots[2] = {0, 1, 1, 1};
            }
            //back neigh is absent
            else if(space(1,1,0).front() >= n_els)
            {
                w_knots[0] = {0, 0, 0, 1};
                w_knots[1] = {0, 0, 1, 2};
                w_knots[2] = {0, 1, 2, 3};
            }
            //front neigh is absent
            else if(space(1,1,2).front() >= n_els)
            {
                w_knots[0] = {-2, -1, 0, 1};
                w_knots[1] = {-1, 0, 1, 1};
                w_knots[2] = {0, 1, 1, 1};
            }
            else
            {
                w_knots[0] = {-2, -1, 0, 1};
                w_knots[1] = {-1, 0, 1, 2};
                w_knots[2] = {0, 1, 2, 3};
            }
        }

        void basis_for_regular_hex(const SpaceMatrix &space, const NodeMatrix &loc_nodes, const std::array<std::vector<double>, 3> &h_knots, const std::array<std::vector<double>, 3> &v_knots, const std::array<std::vector<double>, 3> &w_knots, ElementBases &b)
        {
            int local_index = 0;
            for(int z = 0; z < 3; ++z)
            {
                for(int y = 0; y < 3; ++y)
                {
                    for(int x = 0; x < 3; ++x)
                    {
                        if(space(x, y, z).size() == 1)
                        {
                            const int global_index = space(x, y, z).front();
                            const Eigen::MatrixXd &node = loc_nodes(x, y, z).front();

                            b.bases[local_index].init(global_index, local_index, node);

                            const QuadraticBSpline3d spline(h_knots[x], v_knots[y], w_knots[z]);

                            // if(global_index == 0)
                            // {
                            //     std::cout<<x<<" "<<y<<" "<<z<<std::endl;
                            //     std::cout<<h_knots[x][0]<<" "<<h_knots[x][1]<<" "<<h_knots[x][2]<<" "<<h_knots[x][3]<<std::endl;
                            //     std::cout<<v_knots[y][0]<<" "<<v_knots[y][1]<<" "<<v_knots[y][2]<<" "<<v_knots[y][3]<<std::endl;
                            //     std::cout<<w_knots[z][0]<<" "<<w_knots[z][1]<<" "<<w_knots[z][2]<<" "<<w_knots[z][3]<<std::endl;
                            // }

                            b.bases[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
                            b.bases[local_index].set_grad( [spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });

                            ++local_index;
                        }
                    }
                }
            }
        }


        void basis_for_q2(const Mesh3D &mesh, const std::vector<ElementType> &els_tag, const int el_index, std::map<int, int > &vertex_id, std::map<int, int > &edge_id, const SpaceMatrix &space, const NodeMatrix &loc_nodes, ElementBases &b, std::map<int, InterfaceData> &poly_edge_to_data, std::vector< int > &bounday_nodes, int &n_bases)
        {
        //     static const auto is_q2 = [els_tag](const int face_id){ return els_tag[face_id] == ElementType::MultiSingularInteriorCube || els_tag[face_id] == ElementType::MultiSingularBoundaryCube; };
        //     const int n_els = mesh.n_elements();

        //     b.bases.resize(9);

        //     Navigation::Index index = mesh.get_index_from_face(el_index);
        //     for (int j = 0; j < 4; ++j)
        //     {
        //         int current_vertex_node_id = -1;
        //         int current_edge_node_id = -1;
        //         Eigen::Matrix<double, 1, 2> current_edge_node;
        //         Eigen::MatrixXd current_vertex_node;


        //         int right_node_id, bottom_right_node_id, bottom_node_id, bottom_left_node_id, left_node_id;
        //         Eigen::MatrixXd right_node, bottom_right_node, bottom_node, bottom_left_node, left_node;

        //         if(j == 1)
        //         {
        //             right_node_id          = space(0, 1).front();
        //             bottom_right_node_id   = space(0, 0).front();
        //             bottom_node_id         = space(1, 0).front();
        //             bottom_left_node_id    = space(2, 0).front();
        //             left_node_id           = space(2, 1).front();

        //             right_node          = loc_nodes(0, 1).front();
        //             bottom_right_node   = loc_nodes(0, 0).front();
        //             bottom_node         = loc_nodes(1, 0).front();
        //             bottom_left_node    = loc_nodes(2, 0).front();
        //             left_node           = loc_nodes(2, 1).front();
        //         }
        //         else if( j == 2)
        //         {
        //             right_node_id          = space(1, 0).front();
        //             bottom_right_node_id   = space(2, 0).front();
        //             bottom_node_id         = space(2, 1).front();
        //             bottom_left_node_id    = space(2, 2).front();
        //             left_node_id           = space(1, 2).front();

        //             right_node          = loc_nodes(1, 0).front();
        //             bottom_right_node   = loc_nodes(2, 0).front();
        //             bottom_node         = loc_nodes(2, 1).front();
        //             bottom_left_node    = loc_nodes(2, 2).front();
        //             left_node           = loc_nodes(1, 2).front();
        //         }
        //         else if( j == 3)
        //         {
        //             right_node_id          = space(2, 1).front();
        //             bottom_right_node_id   = space(2, 2).front();
        //             bottom_node_id         = space(1, 2).front();
        //             bottom_left_node_id    = space(0, 2).front();
        //             left_node_id           = space(0, 1).front();

        //             right_node          = loc_nodes(2, 1).front();
        //             bottom_right_node   = loc_nodes(2, 2).front();
        //             bottom_node         = loc_nodes(1, 2).front();
        //             bottom_left_node    = loc_nodes(0, 2).front();
        //             left_node           = loc_nodes(0, 1).front();
        //         }
        //         else
        //         {
        //             right_node_id          = space(1, 2).front();
        //             bottom_right_node_id   = space(0, 2).front();
        //             bottom_node_id         = space(0, 1).front();
        //             bottom_left_node_id    = space(0, 0).front();
        //             left_node_id           = space(1, 0).front();

        //             right_node          = loc_nodes(1, 2).front();
        //             bottom_right_node   = loc_nodes(0, 2).front();
        //             bottom_node         = loc_nodes(0, 1).front();
        //             bottom_left_node    = loc_nodes(0, 0).front();
        //             left_node           = loc_nodes(1, 0).front();
        //         }

        //         const int opposite_face = mesh.switch_face(index).face;
        //         const int other_face = mesh.switch_face(mesh.switch_edge(index)).face;
        //         const bool is_vertex_q2 = other_face < 0 || mesh.n_element_vertices(other_face) > 4 || is_q2(other_face);

        //         if (opposite_face < 0 || mesh.n_element_vertices(opposite_face) > 4)
        //         {
        //             auto eit = edge_id.find(index.edge);

        //             if(eit == edge_id.end())
        //             {
        //                 current_edge_node_id = ++n_bases;
        //                 edge_id[index.edge] = current_edge_node_id;
        //             }
        //             else
        //                 current_edge_node_id = eit->second;

        //             current_edge_node = mesh.edge_mid_point(index.edge);
        //             if(opposite_face < 0)
        //                 bounday_nodes.push_back(current_edge_node_id);

        //             if(is_vertex_q2)
        //             {
        //                 auto vit = vertex_id.find(index.vertex);

        //                 if(vit == vertex_id.end())
        //                 {
        //                     current_vertex_node_id = ++n_bases;
        //                     vertex_id[index.vertex] = current_vertex_node_id;
        //                 }
        //                 else
        //                     current_vertex_node_id = vit->second;

        //                 if(opposite_face < 0)
        //                     bounday_nodes.push_back(current_vertex_node_id);
        //                 mesh.point(index.vertex, current_vertex_node);
        //             }
        //         }
        //         else
        //         {
        //             const bool is_edge_q2 = is_q2(opposite_face);

        //             if(is_edge_q2)
        //             {
        //                 if(is_vertex_q2)
        //                 {
        //                     auto it = vertex_id.find(index.vertex);

        //                     if(it == vertex_id.end())
        //                     {
        //                         current_vertex_node_id = ++n_bases;
        //                         vertex_id[index.vertex] = current_vertex_node_id;
        //                     }
        //                     else
        //                         current_vertex_node_id = it->second;

        //                     if(other_face < 0)
        //                         bounday_nodes.push_back(current_vertex_node_id);
        //                     mesh.point(index.vertex, current_vertex_node);
        //                 }

        //                 auto it = edge_id.find(index.edge);

        //                 if(it == edge_id.end())
        //                 {
        //                     current_edge_node_id = ++n_bases;
        //                     edge_id[index.edge] = current_edge_node_id;
        //                 }
        //                 else
        //                     current_edge_node_id = it->second;

        //                 current_edge_node = mesh.edge_mid_point(index.edge);
        //             }
        //         }


        //         // const bool is_neigh_poly = (opposite_face >= 0 && mesh.n_element_vertices(opposite_face) > 4);
        //         // const bool is_other_neigh_poly = (other_face >= 0 && mesh.n_element_vertices(other_face) > 4);

        //         if(current_vertex_node_id >= 0)
        //         {
        //             b.bases[2*j].init(current_vertex_node_id, 2*j, current_vertex_node);
        //         }
        //         else
        //         {
        //             // std::cout<<j<<" n "<<right_node_id<< " "<<bottom_right_node_id<< " "<<bottom_node_id<<" "<<bottom_left_node_id<< " "<<left_node_id<<std::endl;

        //             auto &global = b.bases[2*j].global();

        //             //central
        //             if(bottom_right_node_id < n_els)
        //                 global.push_back(Local2Global(el_index, mesh.node_from_face(el_index), 1./4.));


        //             if(bottom_right_node_id < n_els)
        //                 global.push_back(Local2Global(right_node_id, right_node, 1./4.));
        //             else if(right_node_id >= n_els)
        //                 global.push_back(Local2Global(right_node_id, right_node, 1./2.));


        //             if(bottom_right_node_id >= n_els)
        //                 global.push_back(Local2Global(bottom_right_node_id, bottom_right_node, 1./2.));
        //             else
        //                 global.push_back(Local2Global(bottom_right_node_id, bottom_right_node, 1./4.));


        //             if(bottom_right_node_id < n_els)
        //                 global.push_back(Local2Global(bottom_node_id, bottom_node, 1./4.));
        //             else if(bottom_node_id >= n_els)
        //                 global.push_back(Local2Global(bottom_node_id, bottom_node, 1./2.));
        //         }


        //         if(current_edge_node_id >= 0)
        //         {
        //             b.bases[2*j+1].init(current_edge_node_id, 2*j+1, current_edge_node);
        //         }
        //         else
        //         {
        //             // std::cout<<j<<" e "<<right_node_id<< " "<<bottom_right_node_id<< " "<<bottom_node_id<<" "<<bottom_left_node_id<< " "<<left_node_id<<std::endl;

        //             auto &global = b.bases[2*j+1].global();

        //             //central
        //             if(right_node_id >= n_els || left_node_id >= n_els)
        //                 global.push_back(Local2Global(el_index, mesh.node_from_face(el_index), 5./16.));
        //             else
        //                 global.push_back(Local2Global(el_index, mesh.node_from_face(el_index), 3./8.));


        //             if(right_node_id >= n_els)
        //                 global.push_back(Local2Global(right_node_id, right_node, 1./8.));
        //             else
        //                 global.push_back(Local2Global(right_node_id, right_node, 1./16.));


        //             if(right_node_id >= n_els)
        //                 global.push_back(Local2Global(bottom_right_node_id, bottom_right_node, 1./8.));
        //             else
        //                 global.push_back(Local2Global(bottom_right_node_id, bottom_right_node, 1./16.));


        //             if(right_node_id >= n_els || left_node_id >= n_els)
        //                 global.push_back(Local2Global(bottom_node_id, bottom_node, 5./16.));
        //             else
        //                 global.push_back(Local2Global(bottom_node_id, bottom_node, 3./8.));


        //             if(left_node_id >= n_els)
        //                 global.push_back(Local2Global(bottom_left_node_id, bottom_left_node, 1./8.));
        //             else
        //                 global.push_back(Local2Global(bottom_left_node_id, bottom_left_node, 1./16.));


        //             if(left_node_id >= n_els)
        //                 global.push_back(Local2Global(left_node_id, left_node, 1./8.));
        //             else
        //                 global.push_back(Local2Global(left_node_id, left_node, 1./16.));
        //         }

        //         const int nj = (j+3)%4;

        //         b.bases[2*j].set_basis([nj](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis2d::quad_basis_basis(2, 2*nj, uv, val); });
        //         b.bases[2*j].set_grad( [nj](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  FEBasis2d::quad_basis_grad(2, 2*nj, uv, val); });

        //         b.bases[2*j+1].set_basis([nj](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis2d::quad_basis_basis(2, 2*nj+1, uv, val); });
        //         b.bases[2*j+1].set_grad( [nj](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  FEBasis2d::quad_basis_grad(2, 2*nj+1, uv, val); });

        //         index = mesh.next_around_face(index);
        //     }

        //     b.bases[8].init(++n_bases, 8, mesh.node_from_face(el_index));
        //     b.bases[8].set_basis([](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis2d::quad_basis_basis(2, 8, uv, val); });
        //     b.bases[8].set_grad( [](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {  FEBasis2d::quad_basis_grad(2, 8, uv, val); });


        //     index = mesh.get_index_from_face(el_index);
        //     for (int j = 0; j < 4; ++j)
        //     {
        //         const int opposite_face = mesh.switch_face(index).face;
        //         const bool is_neigh_poly = (opposite_face >= 0 && mesh.n_element_vertices(opposite_face) > 4);

        //         int b_flag;

        //         if(j == 1)
        //             b_flag = InterfaceData::TOP_FLAG;
        //         else if( j == 2)
        //             b_flag = InterfaceData::LEFT_FLAG;
        //         else if( j == 3)
        //             b_flag = InterfaceData::BOTTOM_FLAG;
        //         else
        //             b_flag = InterfaceData::RIGHT_FLAG;

        //         if(is_neigh_poly)
        //         {
        //             InterfaceData &data = poly_edge_to_data[index.edge];
        //             data.face_id = index.face;
        //             data.flag = b_flag;

        //             auto &bases_e = b.bases[2*j+1];
        //             for(std::size_t i = 0; i < bases_e.global().size(); ++i)
        //             {
        //                 data.node_id.push_back(bases_e.global()[i].index);
        //                 data.local_indices.push_back(2*j+1);
        //                 data.vals.push_back(bases_e.global()[i].val);
        //             }

        //             auto &bases_v1 = b.bases[2*j];
        //             for(std::size_t i = 0; i < bases_v1.global().size(); ++i)
        //             {
        //                 data.node_id.push_back(bases_v1.global()[i].index);
        //                 data.local_indices.push_back(2*j);
        //                 data.vals.push_back(bases_v1.global()[i].val);
        //             }

        //             const int ii = (2*j+2) >= 8 ? 0 : (2*j+2);
        //             auto &bases_v2 = b.bases[ii];
        //             for(std::size_t i = 0; i < bases_v2.global().size(); ++i)
        //             {
        //                 data.node_id.push_back(bases_v2.global()[i].index);
        //                 data.local_indices.push_back(ii);
        //                 data.vals.push_back(bases_v2.global()[i].val);
        //             }
        //         }

        //         index = mesh.next_around_face(index);
        //     }
        }
    }


    int SplineBasis3d::build_bases(const Mesh3D &mesh, const std::vector<ElementType> &els_tag, const int quadrature_order, std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes, std::map<int, Eigen::MatrixXd> &polys)
    {
        using std::max;
        assert(mesh.is_volume());

        const int n_els = mesh.n_elements();
        bases.resize(n_els);
        local_boundary.resize(n_els);

        bounday_nodes.clear();

        int n_bases = n_els;

        HexQuadrature hex_quadrature;

        std::map<int, InterfaceData> poly_edge_to_data;

        std::array<std::vector<double>, 3> h_knots;
        std::array<std::vector<double>, 3> v_knots;
        std::array<std::vector<double>, 3> w_knots;

        for(int e = 0; e < n_els; ++e)
        {
            if(els_tag[e] != ElementType::RegularInteriorCube && els_tag[e] != ElementType::RegularBoundaryCube && els_tag[e] != ElementType::SimpleSingularInteriorCube && els_tag[e] != ElementType::SimpleSingularBoundaryCube)
                continue;

            //needs to be implemented
            assert(els_tag[e] != ElementType::SimpleSingularInteriorCube && els_tag[e] != ElementType::SimpleSingularBoundaryCube);

            SpaceMatrix space;
            NodeMatrix loc_nodes;

            const int max_local_base = build_local_space(mesh, e, space, loc_nodes, local_boundary[e], poly_edge_to_data, bounday_nodes);
            n_bases = max(n_bases, max_local_base);

            ElementBases &b=bases[e];
            hex_quadrature.get_quadrature(quadrature_order, b.quadrature);
            b.bases.resize(27);


            setup_knots_vectors(n_els, space, h_knots, v_knots, w_knots);
            // print_local_space(space);

            basis_for_regular_hex(space, loc_nodes, h_knots, v_knots, w_knots, b);
            // basis_for_irregulard_hex(mesh, space, loc_nodes, h_knots, v_knots, b);
        }


        std::map<int, int > edge_id;
        std::map<int, int > vertex_id;

        for(int e = 0; e < n_els; ++e)
        {
            if(els_tag[e] != ElementType::MultiSingularInteriorCube && els_tag[e] != ElementType::MultiSingularBoundaryCube)
                continue;

            SpaceMatrix space;
            NodeMatrix loc_nodes;
            std::map<int, InterfaceData> dummy;
            build_local_space(mesh, e, space, loc_nodes, local_boundary[e], dummy, bounday_nodes);
            ElementBases &b=bases[e];
            hex_quadrature.get_quadrature(quadrature_order, b.quadrature);

            basis_for_q2(mesh, els_tag, e, vertex_id, edge_id, space, loc_nodes, b, poly_edge_to_data, bounday_nodes, n_bases);
            std::cout<<b<<std::endl;

        }

        return n_bases+1;
    }

}

#include "SplineBasis3d.hpp"

#include "QuadraticBSpline3d.hpp"
#include "HexQuadrature.hpp"

#include "HexBoundarySampler.hpp"
#include "FEBasis3d.hpp"



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

        bool is_edge_singular(const Navigation3D::Index &index, const Mesh3D &mesh)
        {
            std::vector<int> ids;
            mesh.get_edge_elements_neighs(index.edge, ids);
            return ids.size() != 4 && !mesh.is_boundary_edge(index.edge);
        }


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

            std::vector<int> ids;
            mesh.get_edge_elements_neighs(index.edge, ids);

            if(is_edge_singular(index, mesh))
            {
                //irregular edge
                space(x, y, z).push_back(index.edge);
                node(x, y, z).push_back(mesh.node_from_edge_index(index));
            }

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
            index = to_face[0](start_index);
            explore_face(index, mesh, 1, 1, 2, TOP_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_face[1](start_index);
            explore_face(index, mesh, 1, 1, 0, BOTTOM_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_face[3](start_index);
            explore_face(index, mesh, 0, 1, 1, LEFT_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_face[2](start_index);
            explore_face(index, mesh, 2, 1, 1, RIGHT_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_face[5](start_index);
            explore_face(index, mesh, 1, 0, 1, FRONT_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_face[4](start_index);
            explore_face(index, mesh, 1, 2, 1, BACK_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);


            ///////////////////////
            index = to_edge[0](start_index);
            explore_edge(index, mesh, 1, 0, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[1](start_index);
            explore_edge(index, mesh, 2, 1, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[2](start_index);
            explore_edge(index, mesh, 1, 2, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[3](start_index);
            explore_edge(index, mesh, 0, 1, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);


            index = to_edge[4](start_index);
            explore_edge(index, mesh, 0, 0, 1, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[5](start_index);
            explore_edge(index, mesh, 2, 0, 1, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[6](start_index);
            explore_edge(index, mesh, 2, 2, 1, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[7](start_index);
            explore_edge(index, mesh, 0, 2, 1, space, node, local_boundary, poly_edge_to_data, bounday_nodes);



            index = to_edge[8](start_index);
            explore_edge(index, mesh, 1, 0, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[9](start_index);
            explore_edge(index, mesh, 2, 1, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[10](start_index);
            explore_edge(index, mesh, 1, 2, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_edge[11](start_index);
            explore_edge(index, mesh, 0, 1, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);


            ////////////////////////////////////////////////////////////////////////
            index = to_vertex[0](start_index);
            explore_vertex(index, mesh, 0, 0, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_vertex[1](start_index);
            explore_vertex(index, mesh, 2, 0, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_vertex[2](start_index);
            explore_vertex(index, mesh, 2, 2, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_vertex[3](start_index);
            explore_vertex(index, mesh, 0, 2, 0, space, node, local_boundary, poly_edge_to_data, bounday_nodes);



            index = to_vertex[4](start_index);
            explore_vertex(index, mesh, 0, 0, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_vertex[5](start_index);
            explore_vertex(index, mesh, 2, 0, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_vertex[6](start_index);
            explore_vertex(index, mesh, 2, 2, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_vertex[7](start_index);
            explore_vertex(index, mesh, 0, 2, 2, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

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
            for(int z = 0; z < 3; ++z)
            {
                for(int y = 0; y < 3; ++y)
                {
                    for(int x = 0; x < 3; ++x)
                    {
                        if(space(1, y, z).size() <= 1 && space(x, 1, z).size() <= 1 && space(x, y, 1).size() <= 1)
                        {
                            const int local_index = 9*z + 3*y + x;
                            const int global_index = space(x, y, z).front();
                            const Eigen::MatrixXd &node = loc_nodes(x, y, z).front();

                            b.bases[local_index].init(global_index, local_index, node);

                            const QuadraticBSpline3d spline(h_knots[x], v_knots[y], w_knots[z]);

                            b.bases[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
                            b.bases[local_index].set_grad( [spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });
                        }
                    }
                }
            }
        }


        void basis_for_irregulard_hex(const Mesh3D &mesh, const SpaceMatrix &space, const NodeMatrix &loc_nodes, const std::array<std::vector<double>, 3> &h_knots, const std::array<std::vector<double>, 3> &v_knots, const std::array<std::vector<double>, 3> &w_knots, ElementBases &b)
        {
            for(int z = 0; z < 3; ++z)
            {
                for(int y = 0; y < 3; ++y)
                {
                    for(int x = 0; x < 3; ++x)
                    {
                        if(space(1, y, z).size() > 1 || space(x, 1, z).size() > 1 || space(x, y, 1).size() > 1)
                        {
                            const int local_index = 9*z + 3*y + x;


                            int mpx = -1;
                            int mpy = -1;
                            int mpz = -1;

                            int mmx = -1;
                            int mmy = -1;
                            int mmz = -1;

                            int xx = 1;
                            int yy = 1;
                            int zz = 1;

                            int edge_id = -1;
                            int dir = -1;

                            if(space(x, y, 1).size() > 1)
                            {
                                assert(space(x, y, 1).size() == 2);
                                edge_id = space(x, y, 1)[1];


                                mpx = 1;
                                mpy = y;
                                mpz = z;

                                mmx = x;
                                mmy = 1;
                                mmz = z;

                                zz = z;
                                dir = z;
                            }
                            else if(space(x, 1, z).size() > 1)
                            {
                                assert(space(x, 1, z).size() == 2);
                                edge_id = space(x, 1, z)[1];


                                mpx = 1;
                                mpy = y;
                                mpz = z;

                                mmx = x;
                                mmy = y;
                                mmz = 1;

                                yy = y;
                                dir = y;

                            }
                            else if(space(1, y, z).size() > 1)
                            {
                                assert(space(1, y, z).size() == 2);
                                edge_id = space(1, y, z)[1];


                                mpx = x;
                                mpy = y;
                                mpz = 1;

                                mmx = x;
                                mmy = 1;
                                mmz = z;

                                xx = x;
                                dir = x;
                            }
                            else
                                assert(false);

                            const int el_index = b.bases[1*9 + 1*3 + 1].global().front().index;

                            const auto &center = b.bases[zz*9 + yy*3 + xx].global().front();

                            const auto &el1 = b.bases[mpz*9 + mpy*3 + mpx].global().front();
                            const auto &el2 = b.bases[mmz*9 + mmy*3 + mmx].global().front();

                            // std::cout<<"el_index "<<el_index<<std::endl;
                            // std::cout<<"center "<<center.index<<std::endl;
                            // std::cout<<"el1 "<<el1.index<<std::endl;
                            // std::cout<<"el2 "<<el2.index<<std::endl;

                            std::vector<int> ids;
                            std::vector<Eigen::MatrixXd> nds;
                            mesh.get_edge_elements_neighs(el_index, edge_id, dir, ids, nds);

                            if(ids.front() != center.index)
                            {
                                assert(dir != 1);
                                ids.clear();
                                nds.clear();
                                mesh.get_edge_elements_neighs(el_index, edge_id, dir == 2 ? 0 : 2, ids, nds);
                            }

                            assert(ids.front() == center.index);

                            std::vector<int> other_indices;
                            std::vector<Eigen::MatrixXd> other_nodes;
                            for(size_t i = 0; i < ids.size(); ++i)
                            {
                                if(ids[i] != center.index && ids[i] != el1.index && ids[i] != el2.index){
                                    other_indices.push_back(ids[i]);
                                    other_nodes.push_back(nds[i]);
                                }
                            }

                            std::cout<<ids.size()<< " " << other_indices.size()<<std::endl;


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
                                base.global()[3+n].node = other_nodes[n];
                            }


                            const QuadraticBSpline3d spline(h_knots[x], v_knots[y], w_knots[z]);

                            b.bases[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
                            b.bases[local_index].set_grad( [spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });
                        }
                    }
                }
            }
        }


        void create_q2_nodes(const Mesh3D &mesh, const std::vector<ElementType> &els_tag, const int el_index, std::set<int> &vertex_id, std::set<int> &edge_id, std::set<int> &face_id, ElementBases &b, std::vector< int > &bounday_nodes, LocalBoundary &local_boundary, int &n_bases)
        {
            const auto is_q2 = [els_tag](const int el_id){ return els_tag[el_id] == ElementType::MultiSingularInteriorCube || els_tag[el_id] == ElementType::MultiSingularBoundaryCube; };
            const int n_els = mesh.n_elements();

            b.bases.resize(27);

            std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> to_face;
            mesh.to_face_functions(to_face);

            std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 12> to_edge;
            mesh.to_edge_functions(to_edge);

            std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 8> to_vertex;
            mesh.to_vertex_functions(to_vertex);

            const Navigation3D::Index start_index = mesh.get_index_from_element(el_index);
            for (int j = 0; j < 8; ++j)
            {
                const Navigation3D::Index index = to_vertex[j](start_index);
                const int loc_index = FEBasis3d::quadr_hex_face_local_nodes(mesh, index)[0];


                int current_vertex_node_id = -1;
                Eigen::MatrixXd current_vertex_node;

                //if the edge/vertex is boundary the it is a Q2 edge
                bool is_vertex_q2 = true;

                std::vector<int> vertex_neighs;
                mesh.get_vertex_elements_neighs(index.vertex, vertex_neighs);

                for(size_t i = 0; i < vertex_neighs.size(); ++i)
                {
                    if(!is_q2(vertex_neighs[i]) && mesh.n_element_vertices(vertex_neighs[i]) == 8)
                    {
                        is_vertex_q2 = false;
                        break;
                    }
                }
                const bool is_vertex_boundary = mesh.is_boundary_vertex(index.vertex);

                if(is_vertex_q2)
                {
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
                    b.bases[loc_index].init(current_vertex_node_id, loc_index, current_vertex_node);

                b.bases[loc_index].set_basis([loc_index](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis3d::quadr_hex_basis_value(loc_index, uv, val); });
                b.bases[loc_index].set_grad( [loc_index](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis3d::quadr_hex_basis_grad(loc_index, uv, val); });
            }


            for (int j = 0; j < 12; ++j)
            {
                Navigation3D::Index index = to_edge[j](start_index);

                int current_edge_node_id = -1;
                Eigen::Matrix<double, 1, 3> current_edge_node;
                const int loc_index = FEBasis3d::quadr_hex_face_local_nodes(mesh, index)[1];


                bool is_edge_q2 = true;

                std::vector<int> edge_neighs;
                mesh.get_edge_elements_neighs(index.edge, edge_neighs);

                for(size_t i = 0; i < edge_neighs.size(); ++i)
                {
                    if(!is_q2(edge_neighs[i]) && mesh.n_element_vertices(edge_neighs[i]) == 8)
                    {
                        is_edge_q2 = false;
                        break;
                    }
                }
                const bool is_edge_boundary = mesh.is_boundary_edge(index.edge);

                if(is_edge_q2)
                {
                    const bool is_new_edge = edge_id.insert(index.edge).second;

                    if(is_new_edge)
                    {
                        current_edge_node_id = ++n_bases;
                        current_edge_node = mesh.node_from_edge(index.edge);

                        if(is_edge_boundary)
                            bounday_nodes.push_back(current_edge_node_id);
                    }
                }

                //init new Q2 nodes
                if(current_edge_node_id >= 0)
                    b.bases[loc_index].init(current_edge_node_id, loc_index, current_edge_node);

                b.bases[loc_index].set_basis([loc_index](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis3d::quadr_hex_basis_value(loc_index, uv, val); });
                b.bases[loc_index].set_grad( [loc_index](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis3d::quadr_hex_basis_grad(loc_index, uv, val); });
            }

            for (int j = 0; j < 6; ++j)
            {
                Navigation3D::Index index = to_face[j](start_index);

                int current_face_node_id = -1;

                Eigen::Matrix<double, 1, 3> current_face_node;
                const int opposite_element = mesh.switch_element(index).element;
                const bool is_face_q2 = opposite_element < 0 || mesh.n_element_vertices(opposite_element) > 8 || is_q2(opposite_element);
                const int loc_index = FEBasis3d::quadr_hex_face_local_nodes(mesh, index)[8];


                if (is_face_q2)
                {
                    const bool is_new_face = face_id.insert(index.face).second;

                    if(is_new_face)
                    {
                        current_face_node_id = ++n_bases;
                        current_face_node = mesh.node_from_face(index.face);

                        const int b_index = loc_index - 20;

                        if(opposite_element < 0)
                        {
                            bounday_nodes.push_back(current_face_node_id);

                            switch(b_index)
                            {
                                case 0:
                                local_boundary.set_left_edge_id(index.face); local_boundary.set_left_boundary();
                                break;

                                case 1:
                                local_boundary.set_right_edge_id(index.face); local_boundary.set_right_boundary();
                                break;

                                case 2:
                                local_boundary.set_front_edge_id(index.face); local_boundary.set_front_boundary();
                                break;

                                case 3:
                                local_boundary.set_back_edge_id(index.face); local_boundary.set_back_boundary();
                                break;

                                case 4:
                                local_boundary.set_bottom_edge_id(index.face); local_boundary.set_bottom_boundary();
                                break;

                                case 5:
                                local_boundary.set_top_edge_id(index.face); local_boundary.set_top_boundary();
                                break;

                            }
                        }
                    }
                }

                //init new Q2 nodes
                if(current_face_node_id >= 0)
                    b.bases[loc_index].init(current_face_node_id, loc_index, current_face_node);

                b.bases[loc_index].set_basis([loc_index](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis3d::quadr_hex_basis_value(loc_index, uv, val); });
                b.bases[loc_index].set_grad( [loc_index](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis3d::quadr_hex_basis_grad(loc_index, uv, val); });
            }

            // //central node always present
            b.bases[26].init(++n_bases, 26, mesh.node_from_element(el_index));
            b.bases[26].set_basis([](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis3d::quadr_hex_basis_value(26, uv, val); });
            b.bases[26].set_grad( [](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { FEBasis3d::quadr_hex_basis_grad(26, uv, val); });
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

        void assign_q2_weights(const Mesh3D &mesh, const std::vector<ElementType> &els_tag, const int el_index, std::vector< ElementBases > &bases)
        {
            const auto is_q2 = [els_tag](const int el_id){ return els_tag[el_id] == ElementType::MultiSingularInteriorCube || els_tag[el_id] == ElementType::MultiSingularBoundaryCube; };

            Eigen::MatrixXd eval_p;
            const Navigation3D::Index start_index = mesh.get_index_from_element(el_index);
            ElementBases &b = bases[el_index];

            std::array<std::function<Navigation3D::Index(Navigation3D::Index)>, 6> to_face;
            mesh.to_face_functions(to_face);
            for (int f = 0; f < 6; ++f)
            {
                const Navigation3D::Index index = to_face[f](start_index);
                const int opposite_element = mesh.switch_element(index).element;

                if(opposite_element < 0 || mesh.n_element_vertices(opposite_element) != 8)
                    continue;

                const auto &param_p     = FEBasis3d::quadr_hex_face_local_nodes_coordinates(mesh, mesh.switch_element(index));
                const auto &other_bases = bases[opposite_element];

                const auto &indices     = FEBasis3d::quadr_hex_face_local_nodes(mesh, index);


                // other_bases.eval_geom_mapping(param_p, eval_p);
                // igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
                // if(!is_q2(opposite_element))
                // {
                //     std::cout<<"local_face "<<f<<std::endl;
                //     viewer.data.add_points(eval_p, Eigen::MatrixXd::Constant(1, 3, 0.));

                //     for(int asd =0; asd<9;++asd)
                //         viewer.data.add_label(eval_p.row(asd), std::to_string(asd));
                // }

                for(std::size_t i = 0; i < other_bases.bases.size(); ++i)
                {
                    const auto &other_b = other_bases.bases[i];

                    if(other_b.global().empty()) continue;

                    other_b.basis(param_p, eval_p);
                    assert(eval_p.size() == 9);

                    //basis i of element opposite element is zero on this elements
                    if(eval_p.cwiseAbs().maxCoeff() <= 1e-10)
                        continue;

                    for(std::size_t k = 0; k < other_b.global().size(); ++k)
                    {
                        for(int l = 0; l < 9; ++l)
                        {
                            Local2Global glob = other_b.global()[k];
                            glob.val *= eval_p(l);

                            insert_into_global(glob, b.bases[indices[l]].global());
                        }
                    }
                }
            }

        }

        void setup_data_for_polygons(const Mesh3D &mesh, const int el_index, const ElementBases &b, std::map<int, InterfaceData> &poly_edge_to_data)
        {
            // Navigation::Index index = mesh.get_index_from_face(el_index);
            // for (int j = 0; j < 4; ++j)
            // {
            //     const int opposite_face = mesh.switch_face(index).face;
            //     const bool is_neigh_poly = (opposite_face >= 0 && mesh.n_element_vertices(opposite_face) > 4);

            //     if(is_neigh_poly)
            //     {
            //         int b_flag;

            //         if(j == 1)
            //             b_flag = InterfaceData::TOP_FLAG;
            //         else if( j == 2)
            //             b_flag = InterfaceData::LEFT_FLAG;
            //         else if( j == 3)
            //             b_flag = InterfaceData::BOTTOM_FLAG;
            //         else
            //             b_flag = InterfaceData::RIGHT_FLAG;

            //         InterfaceData &data = poly_edge_to_data[index.edge];
            //         data.face_id = index.face;
            //         data.flag = b_flag;

            //         const auto &bases_e = b.bases[2*j+1];
            //         for(std::size_t i = 0; i < bases_e.global().size(); ++i)
            //         {
            //             data.node_id.push_back(bases_e.global()[i].index);
            //             data.local_indices.push_back(2*j+1);
            //             data.vals.push_back(bases_e.global()[i].val);
            //         }

            //         const auto &bases_v1 = b.bases[2*j];
            //         for(std::size_t i = 0; i < bases_v1.global().size(); ++i)
            //         {
            //             data.node_id.push_back(bases_v1.global()[i].index);
            //             data.local_indices.push_back(2*j);
            //             data.vals.push_back(bases_v1.global()[i].val);
            //         }

            //         const int ii = (2*j+2) >= 8 ? 0 : (2*j+2);
            //         const auto &bases_v2 = b.bases[ii];
            //         for(std::size_t i = 0; i < bases_v2.global().size(); ++i)
            //         {
            //             data.node_id.push_back(bases_v2.global()[i].index);
            //             data.local_indices.push_back(ii);
            //             data.vals.push_back(bases_v2.global()[i].val);
            //         }
            //     }

            //     index = mesh.next_around_face(index);
            // }
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
            basis_for_irregulard_hex(mesh, space, loc_nodes, h_knots, v_knots, w_knots, b);
        }


        std::set<int> face_id;
        std::set<int> edge_id;
        std::set<int> vertex_id;

        for(int e = 0; e < n_els; ++e)
        {
            if(els_tag[e] != ElementType::MultiSingularInteriorCube && els_tag[e] != ElementType::MultiSingularBoundaryCube)
                continue;

            ElementBases &b=bases[e];
            hex_quadrature.get_quadrature(quadrature_order, b.quadrature);

            create_q2_nodes(mesh, els_tag, e, vertex_id, edge_id, face_id, b, bounday_nodes, local_boundary[e], n_bases);
        }


        for(int e = 0; e < n_els; ++e)
        {
            if(els_tag[e] != ElementType::MultiSingularInteriorCube && els_tag[e] != ElementType::MultiSingularBoundaryCube)
                continue;
            assign_q2_weights(mesh, els_tag, e, bases);
        }

        for(int e = 0; e < n_els; ++e)
        {
            if(els_tag[e] != ElementType::MultiSingularInteriorCube && els_tag[e] != ElementType::MultiSingularBoundaryCube)
                continue;
            assign_q2_weights(mesh, els_tag, e, bases);
        }

        // for(int e = 0; e < n_els; ++e)
        // {
        //     if(els_tag[e] != ElementType::MultiSingularInteriorCube && els_tag[e] != ElementType::MultiSingularBoundaryCube)
        //         continue;
        //     const ElementBases &b=bases[e];
        //     setup_data_for_polygons(mesh, e, b, poly_edge_to_data);
        // }

        return n_bases+1;
    }

}

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

        void add_id_for_poly(const Navigation3D::Index &index, const int x1, const int y1, const int x2, const int y2, const SpaceMatrix &space, std::map<int, InterfaceData> &poly_edge_to_data)
        {
            // auto it = poly_edge_to_data.find(index.edge);
            // if(it != poly_edge_to_data.end())
            // {
            //     InterfaceData &data = it->second;

            //     assert(space(x1, y1).size() == 1);
            //     data.node_id.push_back(space(x1, y1).front());
            //     data.x.push_back(x1);
            //     data.y.push_back(y1);

            //     assert(space(x2, y2).size() == 1);
            //     data.node_id.push_back(space(x2, y2).front());
            //     data.x.push_back(x2);
            //     data.y.push_back(y2);
            // }
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
            explore_face(index, mesh, 1, 0, 1, BACK_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);

            index = to_face[5](start_index);
            explore_face(index, mesh, 1, 2, 1, FRONT_FLAG, space, node, local_boundary, poly_edge_to_data, bounday_nodes);


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

        void basis_for_irregulard_hex(const Mesh3D &mesh, const SpaceMatrix &space, const NodeMatrix &loc_nodes, const std::array<std::vector<double>, 3> &h_knots, const std::array<std::vector<double>, 3> &v_knots, ElementBases &b)
        {
            // for(int y = 0; y < 3; ++y)
            // {
            //     for(int x = 0; x < 3; ++x)
            //     {
            //         if(space(x, y).size() > 1)
            //         {
            //             const int mpx = 1;
            //             const int mpy = y;

            //             const int mmx = x;
            //             const int mmy = 1;

            //             std::vector<int> other_indices;
            //             const auto &center = b.bases[1*3 + 1].global().front();

            //             const auto &el1 = b.bases[mpy*3 + mpx].global().front();
            //             const auto &el2 = b.bases[mmy*3 + mmx].global().front();

            //             Navigation::Index index = mesh.get_index_from_face(center.index);
            //             while(mesh.next_around_vertex(index).face != el1.index && mesh.next_around_vertex(index).face != el2.index)
            //             {
            //                 index = mesh.next_around_face(index);
            //             }

            //             index = mesh.next_around_vertex(index);

            //             Navigation::Index i1 = mesh.next_around_vertex(index);
            //             if(i1.face == space(x,y)[0] || i1.face == space(x,y)[1])
            //                 index = i1;
            //             else
            //                 index = mesh.next_around_vertex(mesh.switch_vertex(index));

            //             const int start = index.face == space(x,y)[0] ? space(x,y)[0] : space(x,y)[1];
            //             const int end = start == space(x,y)[0] ? space(x,y)[1] : space(x,y)[0];
            //             assert(index.face == space(x,y)[0] || index.face == space(x,y)[1]);

            //             while(index.face != end)
            //             {
            //                 other_indices.push_back(index.face);
            //                 index = mesh.next_around_vertex(index);
            //             }
            //             other_indices.push_back(end);


            //             const int local_index = y*3 + x;
            //             auto &base = b.bases[local_index];

            //             const int k = int(other_indices.size()) + 3;


            //             base.global().resize(k);

            //             base.global()[0].index = center.index;
            //             base.global()[0].val = (4. - k) / k;
            //             base.global()[0].node = center.node;

            //             base.global()[1].index = el1.index;
            //             base.global()[1].val = (4. - k) / k;
            //             base.global()[1].node = el1.node;

            //             base.global()[2].index = el2.index;
            //             base.global()[2].val = (4. - k) / k;
            //             base.global()[2].node = el2.node;


            //             for(std::size_t n = 0; n < other_indices.size(); ++n)
            //             {
            //                 base.global()[3+n].index = other_indices[n];
            //                 base.global()[3+n].val = 4./k;
            //                 base.global()[3+n].node = mesh.node_from_face(other_indices[n]);
            //             }


            //             const QuadraticBSpline2d spline(h_knots[x], v_knots[y]);
            //             b.bases[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
            //             b.bases[local_index].set_grad( [spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });
            //         }
            //     }
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
            if(els_tag[e] != ElementType::RegularInteriorCube && els_tag[e] != ElementType::RegularBoundaryCube && els_tag[e] != ElementType::SimpleSingularInteriorCube)
                continue;

            //needs to be implemented
            assert(els_tag[e] != ElementType::SimpleSingularInteriorCube);

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

            // SpaceMatrix space;
            // NodeMatrix loc_nodes;
            // std::map<int, BoundaryData> dummy;
            // build_local_space(mesh, e, space, loc_nodes, local_boundary[e], dummy, bounday_nodes);
            // ElementBases &b=bases[e];
            // quad_quadrature.get_quadrature(quadrature_order, b.quadrature);

            // basis_for_q2(mesh, els_tag, e, vertex_id, edge_id, space, loc_nodes, b, poly_edge_to_data, bounday_nodes, n_bases);
            // std::cout<<b<<std::endl;

        }

        // const int samples_res = 5;
        // const bool use_harmonic = true;
        // const bool c1_continuous = !use_harmonic && true;


        // PolygonQuadrature poly_quad;
        // Eigen::Matrix2d det_mat;
        // Eigen::MatrixXd p0, p1;

        // for(int e = 0; e < n_els; ++e)
        // {
        //     const int n_edges = mesh.n_element_vertices(e);

        //     if(n_edges == 4)
        //         continue;

        //     double area = 0;
        //     for(int i = 0; i < n_edges; ++i)
        //     {
        //         const int ip = (i + 1) % n_edges;

        //         mesh.point(mesh.vertex_global_index(e, i), p0);
        //         mesh.point(mesh.vertex_global_index(e, ip), p1);
        //         det_mat.row(0) = p0;
        //         det_mat.row(1) = p1;

        //         area += det_mat.determinant();
        //     }
        //     area = fabs(area);
        //     // const double eps = use_harmonic ? (0.08*area) : 0;
        //     const double eps = 0.08*area;

        //     std::vector<int> local_to_global;
        //     Eigen::MatrixXd boundary_samples, poly_samples;
        //     Eigen::MatrixXd rhs;

        //     sample_polygon(e, samples_res, mesh, poly_edge_to_data, bases, local_to_global, eps, c1_continuous, boundary_samples, poly_samples, rhs);

        //     ElementBases &b=bases[e];
        //     b.has_parameterization = false;
        //     poly_quad.get_quadrature(boundary_samples, quadrature_order, b.quadrature);

        //     polys[e] = boundary_samples;

        //     const int n_poly_bases = int(local_to_global.size());
        //     b.bases.resize(n_poly_bases);

        //     if(use_harmonic)
        //     {
        //         Harmonic harmonic(poly_samples, boundary_samples, rhs);

        //         // igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
        //         // viewer.data.add_points(poly_samples, Eigen::Vector3d(0,1,1).transpose());

        //         // viewer.data.add_points(boundary_samples, Eigen::Vector3d(1,0,1).transpose());
        //     // for(int asd = 0; asd < boundary_samples.rows(); ++asd)
        //         // viewer.data.add_label(boundary_samples.row(asd), std::to_string(asd));

        //         for(int i = 0; i < n_poly_bases; ++i)
        //         {
        //             b.bases[i].init(local_to_global[i], i, Eigen::MatrixXd(1,2));
        //             b.bases[i].set_basis([harmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { harmonic.basis(i, uv, val); });
        //             b.bases[i].set_grad( [harmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { harmonic.grad(i, uv, val); });
        //         }
        //     }
        //     else
        //     {
        //         Biharmonic biharmonic(poly_samples, boundary_samples, rhs);

        //         for(int i = 0; i < n_poly_bases; ++i)
        //         {
        //             b.bases[i].init(local_to_global[i], i, Eigen::MatrixXd(1,2));
        //             b.bases[i].set_basis([biharmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { biharmonic.basis(i, uv, val); });
        //             b.bases[i].set_grad( [biharmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { biharmonic.grad(i, uv, val); });
        //         }
        //     }
        // }

        return n_bases+1;
    }

}

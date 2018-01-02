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

        struct BoundaryData
        {
            int face_id;
            int flag;
            std::vector<int> node_id;

            std::vector<int> x, y;
        };


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

        void explore_edge(const Navigation3D::Index &index, const Mesh3D &mesh, const int x, const int y, const int z, SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, BoundaryData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
        {
            int node_id;
            bool boundary = mesh.node_id_from_edge_index(index, node_id);

            assert(std::find(space(x, y, z).begin(), space(x, y, z).end(), node_id) == space(x, y, z).end());
            space(x, y, z).push_back(node_id);
            node(x, y, z).push_back(mesh.node_from_edge_index(index));

            if(node_id >= mesh.n_elements() && boundary)
                bounday_nodes.push_back(node_id);
        }

        void explore_vertex(const Navigation3D::Index &index, const Mesh3D &mesh, const int x, const int y, const int z, SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, BoundaryData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
        {
            int node_id;
            bool boundary = mesh.node_id_from_vertex_index(index, node_id);

            assert(std::find(space(x, y, z).begin(), space(x, y, z).end(), node_id) == space(x, y, z).end());
            space(x, y, z).push_back(node_id);
            node(x, y, z).push_back(mesh.node_from_vertex_index(index));

            if(node_id >= mesh.n_elements() && boundary)
                bounday_nodes.push_back(node_id);
        }

        void explore_face(const Navigation3D::Index &index, const Mesh3D &mesh, const int x, const int y, const int z,  const int b_flag, SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, BoundaryData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
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

        

        void add_id_for_poly(const Navigation3D::Index &index, const int x1, const int y1, const int x2, const int y2, const SpaceMatrix &space, std::map<int, BoundaryData> &poly_edge_to_data)
        {
            // auto it = poly_edge_to_data.find(index.edge);
            // if(it != poly_edge_to_data.end())
            // {
            //     BoundaryData &data = it->second;

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

        int build_local_space(const Mesh3D &mesh, const int el_index,  SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, BoundaryData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
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

            std::cout<<std::endl;
            print_local_space(space);




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
                        // assert(space(i,j,l).size() >= 1);
                        for(std::size_t k = 0; k < space(i,j,l).size(); ++k)
                        {
                            minCoeff = std::min(space(i,j,l)[k], minCoeff);
                            maxCoeff = std::max(space(i,j,l)[k], maxCoeff);
                        }
                    }
                }
            }

            // assert(minCoeff >= 0);
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

        void sample_polygon(const int element_index, const int samples_res, const Mesh3D &mesh, std::map<int, BoundaryData> &poly_edge_to_data, const std::vector< ElementBases > &bases, std::vector<int> &local_to_global, const double eps, const bool c1_continuous, Eigen::MatrixXd &boundary_samples, Eigen::MatrixXd &poly_samples, Eigen::MatrixXd &rhs)
        {
            // const int n_edges = mesh.n_element_vertices(element_index);

            // const int poly_local_n = (samples_res - 1)/3;
            // const int n_samples      = (samples_res - 1) * n_edges;
            // const int n_poly_samples = poly_local_n * n_edges;

            // boundary_samples.resize(n_samples, 2);
            // poly_samples.resize(n_poly_samples, 2);

            // Eigen::MatrixXd samples, mapped, basis_val, grad_basis_val;
            // std::vector<Eigen::MatrixXd> grads;

            // Navigation::Index index = mesh.get_index_from_face(element_index);
            // for(int i = 0; i < n_edges; ++i)
            // {
            //     const BoundaryData &bdata = poly_edge_to_data[index.edge];
            //     local_to_global.insert(local_to_global.end(), bdata.node_id.begin(), bdata.node_id.end());

            //     index = mesh.next_around_face(index);
            // }

            // std::sort( local_to_global.begin(), local_to_global.end() );
            // local_to_global.erase( std::unique( local_to_global.begin(), local_to_global.end() ), local_to_global.end() );
            // // assert(int(local_to_global.size()) <= n_edges);

            // rhs = Eigen::MatrixXd::Zero(n_samples + (c1_continuous? (2*n_samples): 0), local_to_global.size());

            // index = mesh.get_index_from_face(element_index);

            // Eigen::MatrixXd prev; //TODO compute first prev!

            // for(int i = 0; i < n_edges; ++i)
            // {
            //     //no boundary polygons
            //     assert(mesh.switch_face(index).face >= 0);

            //     const BoundaryData &bdata = poly_edge_to_data[index.edge];
            //     const ElementBases &b=bases[bdata.face_id];
            //     assert(bdata.face_id == mesh.switch_face(index).face);

            //     QuadBoundarySampler::sample(bdata.flag == RIGHT_FLAG, bdata.flag == BOTTOM_FLAG, bdata.flag == LEFT_FLAG, bdata.flag == TOP_FLAG, samples_res, false, samples);

            //     b.eval_geom_mapping(samples, mapped);

            //     if(c1_continuous)
            //     {
            //         b.eval_geom_mapping_grads(samples, grads);
            //     }

            //     bool must_reverse = true;
            //     if(prev.size() > 0)
            //     {
            //         const double dist_first = (mapped.row(0)-prev).norm();

            //         if(dist_first < 1e-8)
            //         {
            //             samples = samples.block(1, 0, samples.rows()-1, samples.cols());
            //             mapped = mapped.block(1, 0, mapped.rows()-1, mapped.cols());

            //             must_reverse = false;
            //         }
            //         else
            //         {
            //             assert((mapped.row(mapped.rows()-1) - prev).norm() < 1e-8);

            //             samples = samples.block(0, 0, samples.rows()-1, samples.cols());
            //             mapped = mapped.block(0, 0, mapped.rows()-1, mapped.cols());

            //             mapped = mapped.colwise().reverse().eval();
            //         }
            //     }
            //     else
            //     {
            //         samples = samples.block(0, 0, samples.rows()-1, samples.cols());
            //         mapped = mapped.block(0, 0, mapped.rows()-1, mapped.cols());

            //         mapped = mapped.colwise().reverse().eval();
            //     }

            //     assert(bdata.node_id.size() == 3);
            //     for(std::size_t bi = 0; bi < bdata.node_id.size(); ++bi)
            //     {
            //         const int local_index = bdata.y[bi] * 3 + bdata.x[bi];
            //         // assert(b.bases[local_index].global_index() == bdata.node_id[bi]);
            //         const long basis_index = std::distance(local_to_global.begin(), std::find(local_to_global.begin(), local_to_global.end(), bdata.node_id[bi]));

            //         b.bases[local_index].basis(samples, basis_val);

            //         if(must_reverse)
            //             basis_val = basis_val.reverse().eval();
            //         rhs.block(i*(samples_res-1), basis_index, basis_val.rows(), 1) = basis_val;

            //         if(c1_continuous)
            //         {
            //             b.bases[local_index].grad(samples, grad_basis_val);

            //             if(must_reverse)
            //                 grad_basis_val = grad_basis_val.colwise().reverse().eval();

            //             for(long k = 0; k < grad_basis_val.rows(); ++k)
            //             {
            //                 const Eigen::MatrixXd trans_grad = grad_basis_val.row(k) * grads[k];

            //                 rhs(n_samples + 2*i*(samples_res-1) + 2*k,     basis_index) = trans_grad(0);
            //                 rhs(n_samples + 2*i*(samples_res-1) + 2*k + 1, basis_index) = trans_grad(1);
            //             }
            //         }

            //     }


            //     prev = mapped.row(mapped.rows()-1);
            //     boundary_samples.block(i*(samples_res-1), 0, mapped.rows(), mapped.cols()) = mapped;
            //     const int offset = int(mapped.rows())/(poly_local_n+1);
            //     for(int j = 0; j < poly_local_n; ++j)
            //     {
            //         const int poly_index = (j+1)*offset-1;

            //         if(eps > 0)
            //         {
            //             const int im = poly_index - 1;
            //             const int ip = poly_index + 1;

            //             const Eigen::MatrixXd e0 = (mapped.row(poly_index) - mapped.row(im)).normalized();
            //             const Eigen::MatrixXd e1 = (mapped.row(ip) - mapped.row(poly_index)).normalized();

            //             const Eigen::Vector2d n0(e0(1), -e0(0));
            //             const Eigen::Vector2d n1(e1(1), -e1(0));
            //             const Eigen::Vector2d n = (n0+n1).normalized(); //TODO discad point if inside

            //             poly_samples.row(i*poly_local_n+j) = n.transpose()*eps + mapped.row(poly_index);
            //         }
            //         else
            //             poly_samples.row(i*poly_local_n+j) = mapped.row(poly_index);
            //     }

            //     index = mesh.next_around_face(index);
            // }
        }
    }


    int SplineBasis3d::build_bases(const Mesh3D &mesh, const int quadrature_order, std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes, std::map<int, Eigen::MatrixXd> &polys)
    {
        using std::max;
        assert(mesh.is_volume());

        const int n_els = mesh.n_elements();
        bases.resize(n_els);
        local_boundary.resize(n_els);

        bounday_nodes.clear();

        int n_bases = n_els;

        HexQuadrature hex_quadrature;

        std::map<int, BoundaryData> poly_edge_to_data;

        std::array<std::vector<double>, 3> h_knots;
        std::array<std::vector<double>, 3> v_knots;
        std::array<std::vector<double>, 3> w_knots;

        for(int e = 0; e < n_els; ++e)
        {
            // if(mesh.n_element_vertices(e) != 8) //TODO use flags
                // continue;

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

#include "Spline2dBasis.hpp"

#include "QuadraticTPBSpline.hpp"
#include "QuadQuadrature.hpp"
#include "PolygonQuadrature.hpp"
#include "QuadBoundarySampler.hpp"
#include "BiharmonicBasis.hpp"

#include <cassert>
#include <iostream>
#include <vector>
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
            int face_id;
            int flag;
            std::vector<int> node_id;

            std::vector<int> x, y;
        };


        void print_local_space(const Matrix3i &space)
        {
            for(int j=2; j >=0; --j)
            {
                for(int i=0; i < 3; ++i)
                {
                    std::cout<<space(i,j)<<" ";
                }
                std::cout<<std::endl;
            }
        }

        void explore_direction(const Navigation::Index &index, const Mesh &mesh, const int x, const int y, const bool is_x, const bool invert, const int b_flag, SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, BoundaryData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
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

        int build_local_space(const Mesh &mesh, const int el_index,  SpaceMatrix &space, NodeMatrix &node, LocalBoundary &local_boundary, std::map<int, BoundaryData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
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
    }


    int Spline2dBasis::build_bases(const Mesh &mesh, const int quadrature_order, std::vector< ElementBases > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes, std::map<int, Eigen::MatrixXd> &polys)
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
            if(mesh.n_element_vertices(e) != 4)
                continue;

            SpaceMatrix space;
            NodeMatrix loc_nodes;

            const int max_local_base = build_local_space(mesh, e, space, loc_nodes, local_boundary[e], poly_edge_to_data, bounday_nodes);
            n_bases = max(n_bases, max_local_base);

            ElementBases &b=bases[e];
            quad_quadrature.get_quadrature(quadrature_order, b.quadrature);
            b.bases.resize(9);

            std::vector<std::vector<double> > h_knots(3);
            std::vector<std::vector<double> > v_knots(3);

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

            // print_local_space(space);

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

                        const QuadraticTensorProductBSpline spline(h_knots[x], v_knots[y]);
                        b.bases[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
                        b.bases[local_index].set_grad( [spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });
                    }
                }
            }

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

                        const int local_index = y*3 + x;
                        auto &base = b.bases[local_index];

                        base.global().resize(5); //TODO

                        base.global()[0].index = b.bases[1*3 + 1].global().front().index;
                        base.global()[0].val = -1./5;
                        base.global()[0].node = b.bases[1*3 + 1].global().front().node;

                        base.global()[1].index = b.bases[mpy*3 + mpx].global().front().index;
                        base.global()[1].val = -1./5;
                        base.global()[1].node = b.bases[mpy*3 + mpx].global().front().node;

                        base.global()[2].index = b.bases[mmy*3 + mmx].global().front().index;
                        base.global()[2].val = -1./5;
                        base.global()[2].node = b.bases[mmy*3 + mmx].global().front().node;



                        base.global()[3].index = space(x,y)[0];
                        base.global()[3].val = 4./5;
                        base.global()[3].node = mesh.node_from_face(space(x,y)[0]);

                        base.global()[4].index = space(x,y)[1];
                        base.global()[4].val = 4./5;
                        base.global()[4].node = mesh.node_from_face(space(x,y)[1]);


                        const QuadraticTensorProductBSpline spline(h_knots[x], v_knots[y]);
                        b.bases[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
                        b.bases[local_index].set_grad( [spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });
                    }
                }
            }
        }

        const int samples_res = 5;

        PolygonQuadrature poly_quad;

        for(int e = 0; e < n_els; ++e)
        {
            const int n_edges = mesh.n_element_vertices(e);

            if(n_edges == 4)
                continue;

            Eigen::MatrixXd samples, mapped, basis_val;

            const int poly_local_n = (samples_res - 1)/3;
            const int n_samples      = (samples_res - 1) * n_edges;
            const int n_poly_samples = poly_local_n * n_edges;

            Eigen::MatrixXd boundary_samples(n_samples, 2);
            Eigen::MatrixXd poly_samples(n_poly_samples, 2);
            Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n_samples, n_edges);

            std::vector<int> local_to_global;

            Navigation::Index index = mesh.get_index_from_face(e);
            for(int i = 0; i < n_edges; ++i)
            {
                const BoundaryData &bdata = poly_edge_to_data[index.edge];
                local_to_global.insert(local_to_global.end(), bdata.node_id.begin(), bdata.node_id.end());

                index = mesh.next_around_face(index);
            }

            std::sort( local_to_global.begin(), local_to_global.end() );
            local_to_global.erase( std::unique( local_to_global.begin(), local_to_global.end() ), local_to_global.end() );
            assert(int(local_to_global.size()) == n_edges);

            index = mesh.get_index_from_face(e);
            for(int i = 0; i < n_edges; ++i)
            {
                const BoundaryData &bdata = poly_edge_to_data[index.edge];
                const ElementBases &b=bases[bdata.face_id];
                assert(bdata.face_id == mesh.switch_face(index).face);

                QuadBoundarySampler::sample(bdata.flag == RIGHT_FLAG, bdata.flag == BOTTOM_FLAG, bdata.flag == LEFT_FLAG, bdata.flag == TOP_FLAG, samples_res, false, samples);
                samples = samples.block(0, 0, samples.rows()-1, samples.cols());
                b.eval_geom_mapping(samples, mapped);
                mapped = mapped.colwise().reverse().eval();
                boundary_samples.block(i*(samples_res-1), 0, mapped.rows(), mapped.cols()) = mapped;

                const int offset = int(mapped.rows())/(poly_local_n+1);
                for(int j = 0; j < poly_local_n; ++j)
                    poly_samples.row(i*poly_local_n+j) = mapped.row((j+1)*offset-1);

                assert(bdata.node_id.size() == 3);
                for(std::size_t bi = 0; bi < bdata.node_id.size(); ++bi)
                {
                    const int local_index = bdata.y[bi] * 3 + bdata.x[bi];
                    // assert(b.bases[local_index].global_index() == bdata.node_id[bi]);
                    const long basis_index = std::distance(local_to_global.begin(), std::find(local_to_global.begin(), local_to_global.end(), bdata.node_id[bi]));

                    b.bases[local_index].basis(samples, basis_val);

                    basis_val = basis_val.reverse().eval();
                    rhs.block(i*(samples_res-1), basis_index, basis_val.rows(), 1) = basis_val;

                }

                index = mesh.next_around_face(index);
            }

            BiharmonicBasis biharmonic(poly_samples, boundary_samples, rhs);

            ElementBases &b=bases[e];
            b.has_parameterization = false;
            poly_quad.get_quadrature(boundary_samples, quadrature_order, b.quadrature);

            polys[e] = boundary_samples;

            const int n_poly_bases = int(local_to_global.size());
            b.bases.resize(n_poly_bases);


            for(int i = 0; i < n_poly_bases; ++i)
            {
                b.bases[i].init(local_to_global[i], i, Eigen::MatrixXd(1,2));
                b.bases[i].set_basis([biharmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { biharmonic.basis(i, uv, val); });
                b.bases[i].set_grad( [biharmonic, i](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { biharmonic.grad(i, uv, val); });
            }
        }

        return n_bases+1;
    }

}

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
        static const int LEFT_FLAG = 1;
        static const int TOP_FLAG = 2;
        static const int RIGHT_FLAG = 4;
        static const int BOTTOM_FLAG = 8;

        struct BoundaryData
        {
            int face_id = -1;
            int flag;
            int node_id;

            int x, y;
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


        int build_local_space(const Mesh &mesh, const int el_index,  Matrix3i &space, Matrix<MatrixXd, 3, 3> &node, LocalBoundary &local_boundary, std::map<int, BoundaryData> &poly_edge_to_data, std::vector< int > &bounday_nodes)
        {
            bool real_boundary;
            int left, bottom, right, top;

            assert(!mesh.is_volume());

            Navigation::Index index;
            space.setConstant(-1);


            space(1, 1) = el_index;
            node(1, 1) = mesh.node_from_face(el_index);

            //////////////////////////////////////////
            index = mesh.get_index_from_face(el_index);

            real_boundary = mesh.node_id_from_edge_index(index, left);
            space(0, 1) = left;
            node(0, 1) = mesh.node_from_edge_index(index);

            if(left < mesh.n_elements())
            {
                Navigation::Index start_index = mesh.switch_face(index);
                assert(start_index.face == left);
                assert(start_index.vertex == index.vertex);

                Navigation::Index edge1 = mesh.switch_edge(start_index);
                mesh.node_id_from_edge_index(edge1, space(0,2));
                node(0,2) = mesh.node_from_edge_index(edge1);

                Navigation::Index edge2 = mesh.switch_edge(mesh.switch_vertex(start_index));
                mesh.node_id_from_edge_index(edge2, space(0,0));
                node(0,0) = mesh.node_from_edge_index(edge2);
            }
            else
            {
                if(real_boundary)
                {
                    local_boundary.set_right_boundary();
                    bounday_nodes.push_back(left);
                }
                else
                {
                    BoundaryData &data = poly_edge_to_data[index.edge];
                    data.face_id = el_index;
                    data.node_id = left;
                    data.flag = RIGHT_FLAG;
                    data.x = 0;
                    data.y = 1;
                }
            }

            //////////////////////////////////////////
            index = mesh.next_around_face(index);
            real_boundary = mesh.node_id_from_edge_index(index, bottom);
            space(1, 0) = bottom;
            node(1, 0) = mesh.node_from_edge_index(index);

            if(bottom < mesh.n_elements()){
                Navigation::Index start_index = mesh.switch_face(index);
                assert(start_index.face == bottom);
                assert(start_index.vertex == index.vertex);

                Navigation::Index edge1 = mesh.switch_edge(start_index);
                mesh.node_id_from_edge_index(edge1, space(0,0));
                node(0,0) = mesh.node_from_edge_index(edge1);

                Navigation::Index edge2 = mesh.switch_edge(mesh.switch_vertex(start_index));
                mesh.node_id_from_edge_index(edge2, space(2,0));
                node(2,0) = mesh.node_from_edge_index(edge2);
            }
            else
            {
                if(real_boundary)
                {
                    local_boundary.set_top_boundary();
                    bounday_nodes.push_back(bottom);
                }
                else
                {
                    BoundaryData &data = poly_edge_to_data[index.edge];
                    data.face_id = el_index;
                    data.node_id = bottom;
                    data.flag = TOP_FLAG;
                    data.x = 1;
                    data.y = 0;
                }
            }

            //////////////////////////////////////////
            index = mesh.next_around_face(index);
            real_boundary = mesh.node_id_from_edge_index(index, right);
            space(2, 1) = right;
            node(2, 1) = mesh.node_from_edge_index(index);

            if(right < mesh.n_elements())
            {
                Navigation::Index start_index = mesh.switch_face(index);
                assert(start_index.face == right);
                assert(start_index.vertex == index.vertex);

                Navigation::Index edge1 = mesh.switch_edge(start_index);
                mesh.node_id_from_edge_index(edge1, space(2,0));
                node(2,0) = mesh.node_from_edge_index(edge1);

                Navigation::Index edge2 = mesh.switch_edge(mesh.switch_vertex(start_index));
                mesh.node_id_from_edge_index(edge2, space(2,2));
                node(2,2) = mesh.node_from_edge_index(edge2);
            }
            else
            {
                if(real_boundary)
                {
                    local_boundary.set_left_boundary();
                    bounday_nodes.push_back(right);
                }
                else
                {
                    BoundaryData &data = poly_edge_to_data[index.edge];
                    data.face_id = el_index;
                    data.node_id = right;
                    data.flag = LEFT_FLAG;
                    data.x = 2;
                    data.y = 1;
                }
            }

            //////////////////////////////////////////
            index = mesh.next_around_face(index);
            real_boundary = mesh.node_id_from_edge_index(index, top);
            space(1, 2) = top;
            node(1, 2) = mesh.node_from_edge_index(index);

            if(top < mesh.n_elements())
            {
                Navigation::Index start_index = mesh.switch_face(index);
                assert(start_index.face == top);
                assert(start_index.vertex == index.vertex);

                Navigation::Index edge1 = mesh.switch_edge(start_index);
                mesh.node_id_from_edge_index(edge1, space(2,2));
                node(2,2) = mesh.node_from_edge_index(edge1);

                Navigation::Index edge2 = mesh.switch_edge(mesh.switch_vertex(start_index));
                mesh.node_id_from_edge_index(edge2, space(0,2));
                node(0,2) = mesh.node_from_edge_index(edge2);
            }
            else
            {
                if(real_boundary)
                {
                    local_boundary.set_bottom_boundary();
                    bounday_nodes.push_back(top);
                }
                else
                {
                    BoundaryData &data = poly_edge_to_data[index.edge];
                    data.face_id = el_index;
                    data.node_id = top;
                    data.flag = BOTTOM_FLAG;
                    data.x = 1;
                    data.y = 2;
                }
            }

            //////////////////////////////////////////
            if(bottom >= mesh.n_elements() && left >= mesh.n_elements())
            {
                Navigation::Index start_index = mesh.get_index_from_face(el_index);
                start_index = mesh.switch_vertex(start_index);

                space(0,0) = mesh.vertex_node_id(start_index.vertex);
                node(0,0) = mesh.node_from_vertex(start_index.vertex);
            }

            if(top >= mesh.n_elements() && left >= mesh.n_elements())
            {
                Navigation::Index start_index = mesh.get_index_from_face(el_index);
                space(0,2) = mesh.vertex_node_id(start_index.vertex);
                node(0,2) = mesh.node_from_vertex(start_index.vertex);
            }

            if(bottom >= mesh.n_elements() && right >= mesh.n_elements())
            {
                Navigation::Index start_index = mesh.get_index_from_face(el_index);
                start_index = mesh.switch_vertex(mesh.next_around_face(start_index));
                space(2,0) = mesh.vertex_node_id(start_index.vertex);
                node(2,0) = mesh.node_from_vertex(start_index.vertex);
            }

            if(top >= mesh.n_elements() && right >= mesh.n_elements())
            {
                Navigation::Index start_index = mesh.get_index_from_face(el_index);
                start_index = mesh.switch_vertex(mesh.switch_edge(start_index));
                space(2,2) = mesh.vertex_node_id(start_index.vertex);
                node(2,2) = mesh.node_from_vertex(start_index.vertex);
            }

            // std::cout<<std::endl;
            // print_local_space(space);

            assert(space.minCoeff() >= 0);
            return space.maxCoeff();
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

        Matrix3i space;
        Matrix<MatrixXd, 3, 3> loc_nodes;

        int n_bases = n_els;

        QuadQuadrature quad_quadrature;

        std::map<int, BoundaryData> poly_edge_to_data;

        for(int e = 0; e < n_els; ++e)
        {
            if(mesh.n_element_vertices(e) != 4)
                continue;

            const int max_local_base = build_local_space(mesh, e, space, loc_nodes, local_boundary[e], poly_edge_to_data, bounday_nodes);
            n_bases = max(n_bases, max_local_base);

            ElementBases &b=bases[e];
            quad_quadrature.get_quadrature(quadrature_order, b.quadrature);
            b.bases.resize(9);

            std::vector<std::vector<double> > h_knots(3);
            std::vector<std::vector<double> > v_knots(3);

            //left and right neigh are absent
            if(space(0,1) >= n_els && space(2,1) >= n_els)
            {
                h_knots[0] = {0, 0, 0, 1};
                h_knots[1] = {0, 0, 1, 1};
                h_knots[2] = {0, 1, 1, 1};
            }
             //left neigh is absent
            else if(space(0,1) >= n_els)
            {
                h_knots[0] = {0, 0, 0, 1};
                h_knots[1] = {0, 0, 1, 2};
                h_knots[2] = {0, 1, 2, 3};
            }
            //right neigh is absent
            else if(space(2,1) >= n_els)
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
            if(space(1,0) >= n_els && space(1,2) >= n_els)
            {
                v_knots[0] = {0, 0, 0, 1};
                v_knots[1] = {0, 0, 1, 1};
                v_knots[2] = {0, 1, 1, 1};
            }
            //bottom neigh is absent
            else if(space(1,0) >= n_els)
            {
                v_knots[0] = {0, 0, 0, 1};
                v_knots[1] = {0, 0, 1, 2};
                v_knots[2] = {0, 1, 2, 3};
            }
            //top neigh is absent
            else if(space(1,2) >= n_els)
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
                    const int global_index = space(x, y);
                    const Eigen::MatrixXd &node = loc_nodes(x,y);

                    const int local_index = y*3 + x;
                    b.bases[local_index].init(global_index, local_index, node);

                    const QuadraticTensorProductBSpline spline(h_knots[x], v_knots[y]);
                    b.bases[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
                    b.bases[local_index].set_grad( [spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });
                }
            }
        }

        const int samples_res = 10;

        PolygonQuadrature poly_quad;

        for(int e = 0; e < n_els; ++e)
        {
            const int n_edges = mesh.n_element_vertices(e);

            if(n_edges == 4)
                continue;

            Navigation::Index index = mesh.get_index_from_face(e);
            Eigen::MatrixXd samples, mapped, basis_val;

            const int poly_local_n = (samples_res - 1)/3;
            const int n_samples      = (samples_res - 1) * n_edges;
            const int n_poly_samples = poly_local_n * n_edges;

            Eigen::MatrixXd boundary_samples(n_samples, 2);
            Eigen::MatrixXd poly_samples(n_poly_samples, 2);
            Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n_samples, n_edges);

            std::vector<int> local_to_global(n_edges);

            for(int i = 0; i < n_edges; ++i)
            {
                const BoundaryData &bdata = poly_edge_to_data[index.edge];
                const int local_index = bdata.y * 3 + bdata.x;
                const ElementBases &b=bases[bdata.face_id];

                local_to_global[i] = bdata.node_id;

                assert(bdata.face_id == mesh.switch_face(index).face);
                assert(b.bases[local_index].global_index() == bdata.node_id);

                QuadBoundarySampler::sample(bdata.flag == RIGHT_FLAG, bdata.flag == BOTTOM_FLAG, bdata.flag == LEFT_FLAG, bdata.flag == TOP_FLAG, samples_res, false, samples);

                samples = samples.block(0, 0, samples.rows()-1, samples.cols());

                Basis::eval_geom_mapping(b.has_parameterization, samples, b.bases, mapped);
                b.bases[local_index].basis(samples, basis_val);

                mapped = mapped.colwise().reverse().eval();
                basis_val = basis_val.reverse().eval();

                boundary_samples.block(i*(samples_res-1), 0, mapped.rows(), mapped.cols()) = mapped;
                rhs.block(i*(samples_res-1), i, basis_val.rows(), 1) = basis_val;

                const int offset = int(mapped.rows())/(poly_local_n+1);
                for(int j = 0; j < poly_local_n; ++j)
                    poly_samples.row(i*poly_local_n+j) = mapped.row((j+1)*offset-1);

                index = mesh.next_around_face(index);
            }

            BiharmonicBasis biharmonic(poly_samples, boundary_samples, rhs);

            // std::cout<<"poly_samples=[\n"<<poly_samples<<"];"<<std::endl;
            // std::cout<<"boundary_samples=[\n"<<boundary_samples<<"];"<<std::endl;
            // std::cout<<"rhs=[\n"<<rhs<<"];"<<std::endl;


            // igl::viewer::Viewer &viewer = UIState::ui_state().viewer;
            // viewer.data.add_points(boundary_samples, Eigen::MatrixXd::Constant(boundary_samples.rows(), 3, 1));
            // viewer.data.add_points(poly_samples, Eigen::MatrixXd::Constant(poly_samples.rows(), 3, 0.3));
            // for(long asd = 0; asd < boundary_samples.rows(); ++asd)
            //     viewer.data.add_label(boundary_samples.row(asd), std::to_string(asd));

            ElementBases &b=bases[e];
            b.has_parameterization = false;
            poly_quad.get_quadrature(boundary_samples, quadrature_order, b.quadrature);

            polys[e] = boundary_samples;

            const int n_poly_bases = int(local_to_global.size());
            b.bases.resize(n_poly_bases);


            // std::cout<<"pts =[\n "<<b.quadrature.points<<"];"<<std::endl;
            // viewer.data.add_points(b.quadrature.points, Eigen::MatrixXd::Constant(b.quadrature.points.rows(), 3, 0.6));

            // Eigen::MatrixXd asd;
            // biharmonic.basis(2, b.quadrature.points, asd);
            // std::cout<<"asd =[\n "<<asd<<"];"<<std::endl;

            // biharmonic.grad(2, b.quadrature.points, asd);
            // std::cout<<"asdg =[\n "<<asd<<"];"<<std::endl;

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

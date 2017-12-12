#include "Spline2dBasis.hpp"

#include "QuadraticTPBSpline.hpp"
#include "navigation.hpp"

#include <cassert>
#include <iostream>
#include <vector>

namespace poly_fem
{

    using namespace Eigen;

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

    int min_v(const int i1, const int i2, const int i3)
    {
        using std::min;
        return min(i1, min(i2, i3));
    }

    int min_v(const int i1, const int i2)
    {
        using std::min;
        return min(i1, i2);
    }


    int build_local_space(const Mesh &mesh, const int el_index,  Matrix3i &space, Matrix<MatrixXd, 3, 3> &node)
    {
        assert(!mesh.is_volume());

        Navigation::Index index;
        space.setConstant(-1);


        space(1, 1) = el_index;
        node(1, 1) = mesh.node_from_face(el_index);

        //////////////////////////////////////////
        index = mesh.get_index_from_face(el_index);
        const int left = mesh.node_id_from_edge_index(index);
        space(0, 1) = left;
        node(0, 1) = mesh.node_from_edge_index(index);

        if(left < mesh.n_elements())
        {
            Navigation::Index start_index = mesh.switch_face(index);
            assert(start_index.face == left);
            assert(start_index.vertex == index.vertex);

            Navigation::Index edge1 = mesh.switch_edge(start_index);
            space(0,2) = mesh.node_id_from_edge_index(edge1);
            node(0,2) = mesh.node_from_edge_index(edge1);

            Navigation::Index edge2 = mesh.switch_edge(mesh.switch_vertex(start_index));
            space(0,0) = mesh.node_id_from_edge_index(edge2);
            node(0,0) = mesh.node_from_edge_index(edge2);
        }

        //////////////////////////////////////////
        index = mesh.next_around_face(index);
        const int bottom = mesh.node_id_from_edge_index(index);
        space(1, 0) = bottom;
        node(1, 0) = mesh.node_from_edge_index(index);

        if(bottom < mesh.n_elements()){
            Navigation::Index start_index = mesh.switch_face(index);
            assert(start_index.face == bottom);
            assert(start_index.vertex == index.vertex);

            Navigation::Index edge1 = mesh.switch_edge(start_index);
            space(0,0) = mesh.node_id_from_edge_index(edge1);
            node(0,0) = mesh.node_from_edge_index(edge1);

            Navigation::Index edge2 = mesh.switch_edge(mesh.switch_vertex(start_index));
            space(2,0) = mesh.node_id_from_edge_index(edge2);
            node(2,0) = mesh.node_from_edge_index(edge2);
        }

        //////////////////////////////////////////
        index = mesh.next_around_face(index);
        const int right = mesh.node_id_from_edge_index(index);
        space(2, 1) = right;
        node(2, 1) = mesh.node_from_edge_index(index);

        if(right < mesh.n_elements())
        {
            Navigation::Index start_index = mesh.switch_face(index);
            assert(start_index.face == right);
            assert(start_index.vertex == index.vertex);

            Navigation::Index edge1 = mesh.switch_edge(start_index);
            space(2,0) = mesh.node_id_from_edge_index(edge1);
            node(2,0) = mesh.node_from_edge_index(edge1);

            Navigation::Index edge2 = mesh.switch_edge(mesh.switch_vertex(start_index));
            space(2,2) = mesh.node_id_from_edge_index(edge2);
            node(2,2) = mesh.node_from_edge_index(edge2);
        }

        //////////////////////////////////////////
        index = mesh.next_around_face(index);
        const int top = mesh.node_id_from_edge_index(index);
        space(1, 2) = top;
        node(1, 2) = mesh.node_from_edge_index(index);

        if(top < mesh.n_elements())
        {
            Navigation::Index start_index = mesh.switch_face(index);
            assert(start_index.face == top);
            assert(start_index.vertex == index.vertex);

            Navigation::Index edge1 = mesh.switch_edge(start_index);
            space(2,2) = mesh.node_id_from_edge_index(edge1);
            node(2,2) = mesh.node_from_edge_index(edge1);

            Navigation::Index edge2 = mesh.switch_edge(mesh.switch_vertex(start_index));
            space(0,2) = mesh.node_id_from_edge_index(edge2);
            node(0,2) = mesh.node_from_edge_index(edge2);
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


    int Spline2dBasis::build_bases(const Mesh &mesh, std::vector< std::vector<Basis> > &bases, std::vector< LocalBoundary > &local_boundary, std::vector< int > &bounday_nodes)
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

        for(int e = 0; e < n_els; ++e)
        {
            const int max_local_base = build_local_space(mesh, e, space, loc_nodes);
            n_bases = max(n_bases, max_local_base);

            std::vector<Basis> &b=bases[e];
            b.resize(9);

            std::vector<std::vector<double> > h_knots(3);
            std::vector<std::vector<double> > v_knots(3);

            //left and right neigh are absent
            if(space(0,1) >= n_els && space(2,1) >= n_els)
            {
                h_knots[0] = {0, 0, 0, 1};
                h_knots[1] = {0, 0, 1, 1};
                h_knots[2] = {0, 1, 1, 1};

                local_boundary[e].set_left_boundary();
                local_boundary[e].set_right_boundary();
            }
             //left neigh is absent
            else if(space(0,1) >= n_els)
            {
                h_knots[0] = {0, 0, 0, 1};
                h_knots[1] = {0, 0, 1, 2};
                h_knots[2] = {0, 1, 2, 3};

                local_boundary[e].set_right_boundary();
            }
            //right neigh is absent
            else if(space(2,1) >= n_els)
            {
                h_knots[0] = {-2, -1, 0, 1};
                h_knots[1] = {-1, 0, 1, 1};
                h_knots[2] = {0, 1, 1, 1};

                local_boundary[e].set_left_boundary();
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

                local_boundary[e].set_top_boundary();
                local_boundary[e].set_bottom_boundary();
            }
            //bottom neigh is absent
            else if(space(1,0) >= n_els)
            {
                v_knots[0] = {0, 0, 0, 1};
                v_knots[1] = {0, 0, 1, 2};
                v_knots[2] = {0, 1, 2, 3};

                local_boundary[e].set_top_boundary();
            }
            //top neigh is absent
            else if(space(1,2) >= n_els)
            {
                v_knots[0] = {-2, -1, 0, 1};
                v_knots[1] = {-1, 0, 1, 1};
                v_knots[2] = {0, 1, 1, 1};

                local_boundary[e].set_bottom_boundary();
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
                    // std::cout<<node<<std::endl;

                    if(global_index >= n_els)
                        bounday_nodes.push_back(global_index);

                    const int local_index = y*3 + x;
                    b[local_index].init(global_index, local_index, node);

                    const QuadraticTensorProductBSpline spline(h_knots[x], v_knots[y]);
                    b[local_index].set_basis([spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.interpolate(uv, val); });
                    b[local_index].set_grad( [spline](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) { spline.derivative(uv, val); });
                }
            }
        }

        return n_bases+1;
    }

}

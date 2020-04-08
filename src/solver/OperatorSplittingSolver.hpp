#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/State.hpp>

#include <polyfem/AssemblerUtils.hpp>
#include <memory>

#include <igl/AABB.h>
#include <igl/in_element.h>

namespace polyfem
{

    class OperatorSplittingSolver
    {
    public:
        OperatorSplittingSolver(const polyfem::Mesh& mesh, const int shape, const int n_el) : shape(shape), n_el(n_el)
        {
            dim = mesh.dimension();

            if (shape == dim + 1)
            {
                T.resize(n_el, dim + 1);
                for (int e = 0; e < n_el; e++)
                {
                    for (int i = 0; i < shape; i++)
                    {
                        T(e, i) = mesh.cell_vertex_(e, i);
                    }
                }
            }
            else
            {
                assert(dim == 2);
                T.resize(n_el * 2, 3);
                for (int e = 0; e < n_el; e++)
                {
                    for (int i = 0; i < 3; i++)
                    {
                        T(e, i) = mesh.cell_vertex_(e, i);
                        T(e + n_el, i) = mesh.cell_vertex_(e, (i + 2) % 4);
                    }
                }
            }
            V = Eigen::MatrixXd::Zero(mesh.n_vertices(), dim);
            for (int i = 0; i < V.rows(); i++)
            {
                auto p = mesh.point(i);
                for (int d = 0; d < dim; d++)
                {
                    V(i, d) = p(d);
                }
            }
            // to find the cell which a point is in
            tree.init(V, T);

            cell_num = (int)pow(n_el, 1./dim);
            hash_table.resize((int)pow(cell_num, dim));

            mesh.bounding_box(min_domain, max_domain);

            for(int e = 0; e < T.rows(); e++)
            {
                VectorXd min_ = V.row(T(e, 0));
                VectorXd max_ = min_;

                for(int i = 1; i < T.cols(); i++)
                {
                    VectorXd p = V.row(T(e, i));
                    for(int d = 0; d < dim; d++)
                    {
                        if(min_(d) > p(d)) min_(d) = p(d);
                        if(max_(d) < p(d)) max_(d) = p(d);
                    }
                }

                VectorXi min_int(dim), max_int(dim);

                for(int d = 0; d < dim; d++)
                {
                    double temp = cell_num / (max_domain(d) - min_domain(d));
                    min_int(d) = floor((min_(d) - min_domain(d)) * temp);
                    max_int(d) = ceil((max_(d) - min_domain(d)) * temp);

                    if(min_int(d) < 0) 
                        min_int(d) = 0;
                    if(max_int(d) > cell_num)
                        max_int(d) = cell_num;
                }

                for(int x = min_int(0); x < max_int(0); x++)
                {
                    for(int y = min_int(1); y < max_int(1); y++)
                    {
                        if(dim == 2)
                        {
                            int idx = x + y * cell_num;
                            hash_table[idx].push_front(e);
                        }
                        else
                        {
                            for(int z = min_int(2); z < max_int(2); z++)
                            {
                                int idx = x + (y + z * cell_num) * cell_num;
                                hash_table[idx].push_front(e);
                            }
                        }
                    }
                }
            }
        }

        void set_bc(const polyfem::Mesh& mesh, const std::vector<polyfem::LocalBoundary>& local_boundary, const std::vector<int>& bnd_nodes, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, Eigen::MatrixXd& sol, const Eigen::MatrixXd& local_pts, const std::shared_ptr<Problem> problem, const double time)
        {
            for (auto e = local_boundary.begin(); e != local_boundary.end(); e++)
            {
                auto elem = *e;
                int elem_idx = elem.element_id();

                // geometry vertices of element e
                std::vector<RowVectorNd> vert(shape);
                for (int i = 0; i < shape; i++)
                {
                    vert[i] = mesh.point(mesh.cell_vertex_(elem_idx, i));
                }

                ElementAssemblyValues gvals;
                gvals.compute(elem_idx, mesh.is_volume(), local_pts, gbases[elem_idx], gbases[elem_idx]);

                for (int local_idx = 0; local_idx < bases[elem_idx].bases.size(); local_idx++)
                {
                    int global_idx = bases[elem_idx].bases[local_idx].global()[0].index;
                    if (find(bnd_nodes.begin(), bnd_nodes.end(), global_idx) == bnd_nodes.end())
                        continue;

                    Eigen::MatrixXd pos = Eigen::MatrixXd::Zero(1, dim);
                    for (int j = 0; j < shape; j++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            pos(0, d) += gvals.basis_values[j].val(local_idx) * vert[j](d);
                        }
                    }

                    Eigen::MatrixXd val;
                    problem->exact(pos, time, val);

                    for (int d = 0; d < dim; d++)
                    {
                        sol(global_idx * dim + d) = val(d);
                    }
                }
            }
        }

        void projection(const polyfem::Mesh& mesh, int n_bases, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, const std::vector<polyfem::ElementBases>& pressure_bases, const Eigen::MatrixXd& local_pts, Eigen::MatrixXd& pressure, Eigen::MatrixXd& sol)
        {
            Eigen::VectorXd grad_pressure = Eigen::VectorXd::Zero(n_bases * dim);
            Eigen::VectorXi traversed = Eigen::VectorXi::Zero(n_bases);

            ElementAssemblyValues vals;
            for (int e = 0; e < n_el; ++e)
            {
                vals.compute(e, mesh.is_volume(), local_pts, pressure_bases[e], gbases[e]);
                for (int j = 0; j < local_pts.rows(); j++)
                {
                    int global_ = bases[e].bases[j].global()[0].index;
                    for (int i = 0; i < vals.basis_values.size(); i++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            assert(pressure(pressure_bases[e].bases[i].global().size() == 1));
                            grad_pressure(global_ * dim + d) += vals.basis_values[i].grad_t_m(j, d) * pressure(pressure_bases[e].bases[i].global()[0].index);
                        }
                    }
                    traversed(global_)++;
                }
            }
            for (int i = 0; i < traversed.size(); i++)
            {
                for (int d = 0; d < dim; d++)
                {
                    sol(i * dim + d) -= grad_pressure(i * dim + d) / traversed(i);
                }
            }
        }

        void initialize_solution(const polyfem::Mesh& mesh, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, const std::shared_ptr<Problem> problem, Eigen::MatrixXd& sol, const Eigen::MatrixXd& local_pts)
        {
            for (int e = 0; e < n_el; e++)
            {
                // geometry vertices of element e
                std::vector<RowVectorNd> vert(shape);
                for (int i = 0; i < shape; i++)
                {
                    vert[i] = mesh.point(mesh.cell_vertex_(e, i));
                }

                // to compute global position with barycentric coordinate
                ElementAssemblyValues gvals;
                gvals.compute(e, mesh.is_volume(), local_pts, gbases[e], gbases[e]);

                for (int i = 0; i < local_pts.rows(); i++)
                {
                    Eigen::MatrixXd pts = Eigen::MatrixXd::Zero(1, dim);
                    for (int j = 0; j < vert.size(); j++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            pts(0, d) += vert[j](d) * gvals.basis_values[j].val(i);
                        }
                    }
                    Eigen::MatrixXd val;
                    problem->initial_solution(pts, val);
                    int global = bases[e].bases[i].global()[0].index;
                    for (int d = 0; d < dim; d++)
                    {
                        sol(global * dim + d) = val(d);
                    }
                }
            }
        }

        int search_cell(RowVectorNd& pos)
        {
            Eigen::VectorXi pos_int(dim);
            for(int d = 0; d < dim; d++)
            {
                pos_int(d) = floor((pos(d) - min_domain(d)) / (max_domain(d) - min_domain(d)) * cell_num);
                if(pos_int(d) < 0) pos_int(d) = 0;
            }

            int idx = 0, dim_num = 1;
            for(int d = 0; d < dim; d++)
            {
                idx += pos_int(d) * dim_num;
                dim_num *= cell_num;
            }

            const std::list<int>& list = hash_table[idx];
            Eigen::MatrixXd points(dim + 2, dim);
            points.row(dim + 1) = pos;
            for(auto it = list.begin(); it != list.end(); it++)
            {
                for(int i = 0; i <= dim; i++)
                {
                    points.row(i) = V.row(T(*it, i));
                }
                
                Eigen::MatrixXd local_pts;
                barycentric_coordinate(points, local_pts);

                if(local_pts.minCoeff() > -1e-13 && local_pts.sum() < 1 + 1e-13)
                {
                    return *it;
                }
            }
            assert(false);
        }

        void barycentric_coordinate(const Eigen::MatrixXd& points, Eigen::MatrixXd& local_pts)
        {
            local_pts.resize(1, dim);
            Eigen::MatrixXd A = Eigen::MatrixXd::Ones(dim + 1, dim + 1);
            A.block(0, 0, dim + 1, dim) = points.block(0, 0, dim + 1, dim);
            double det = A.determinant();
            assert(det > 0);
            for(int i = 1; i <= dim; i++)
            {
                Eigen::MatrixXd B = A;
                for(int j = 0; j < dim; j++)
                {
                    B(i, j) = points(dim + 1, j);
                }
                local_pts(i - 1) = B.determinant() / det;
            }
        }

        void calculate_local_pts(const polyfem::Mesh& mesh, const polyfem::ElementBases& gbase, const int elem_idx, const RowVectorNd& pos, Eigen::MatrixXd& local_pos)
        {
            local_pos = Eigen::MatrixXd::Zero(1, dim);
            
            std::vector<RowVectorNd> vert(shape);
            for (int i = 0; i < shape; i++)
            {
                vert[i] = mesh.point(mesh.cell_vertex_(elem_idx, i));
            }

            ElementAssemblyValues gvals_;
            gvals_.compute(elem_idx, mesh.is_volume(), local_pos, gbase, gbase);

            Eigen::MatrixXd res = -pos;
            for (int i = 0; i < shape; i++)
            {
                res += vert[i] * gvals_.basis_values[i].val(0);
            }

            Eigen::MatrixXd jacobi = Eigen::MatrixXd::Zero(dim, dim);
            for (int d1 = 0; d1 < dim; d1++)
            {
                for (int d2 = 0; d2 < dim; d2++)
                {
                    for (int i = 0; i < shape; i++)
                    {
                        jacobi(d1, d2) += vert[i](d1) * gvals_.basis_values[i].grad(0, d2);
                    }
                }
            }

            Eigen::VectorXd delta = jacobi.colPivHouseholderQr().solve(res.transpose());
            for (int d = 0; d < dim; d++)
            {
                local_pos(d) -= delta(d);
            }
        }

        void advection(const polyfem::Mesh& mesh, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, Eigen::MatrixXd& sol, const double dt, const Eigen::MatrixXd& local_pts, const bool BFS = false, const int order = 1)
        {
            // to store new velocity
            Eigen::MatrixXd new_sol = Eigen::MatrixXd::Zero(sol.size(), 1);
            // number of FEM nodes
            const int n_vert = sol.size() / dim;
            Eigen::VectorXi traversed = Eigen::VectorXi::Zero(n_vert);

            for (int e = 0; e < n_el; e++)
            {
                // geometry vertices of element e
                std::vector<RowVectorNd> vert(shape);
                for (int i = 0; i < shape; i++)
                {
                    vert[i] = mesh.point(mesh.cell_vertex_(e, i));
                }

                // to compute global position with barycentric coordinate
                ElementAssemblyValues gvals;
                gvals.compute(e, mesh.is_volume(), local_pts, gbases[e], gbases[e]);

                for (int i = 0; i < local_pts.rows(); i++)
                {
                    // global index of this FEM node
                    int global = bases[e].bases[i].global()[0].index;

                    if (traversed(global)) continue;
                    traversed(global) = 1;

                    // velocity of this FEM node
                    RowVectorNd vel(dim);
                    for (int d = 0; d < dim; d++)
                    {
                        vel(d) = sol(global * dim + d);
                    }

                    // global position of this FEM node
                    RowVectorNd pos = RowVectorNd::Zero(1, dim);
                    for (int j = 0; j < shape; j++)
                    {
                        pos += gvals.basis_values[j].val(i) * vert[j];
                    }

                    // trace back
                    pos = pos - vel * dt;

                    // to avoid that pos is out of domain
                    for (int d = 0; d < dim; d++)
                    {
                        if (pos(d) <= min_domain(d)) pos(d) = min_domain(d) + 1e-13;
                        if (pos(d) >= max_domain(d)) pos(d) = max_domain(d) - 1e-13;
                    }

                    Eigen::VectorXi I(1);
                    Eigen::MatrixXd local_pos;
                    
                    if(!BFS)
                    {
                        assert(dim == 2);
                        igl::in_element(V, T, pos, tree, I);
                    }
                    else
                    {
                        I(0) = search_cell(pos);
                    }

                    I(0) = I(0) % n_el;
                    calculate_local_pts(mesh, gbases[I(0)], I(0), pos, local_pos);

                    // interpolation
                    ElementAssemblyValues vals;
                    vals.compute(I(0), mesh.is_volume(), local_pos, bases[I(0)], gbases[I(0)]);
                    for (int d = 0; d < dim; d++)
                    {
                        for (int i = 0; i < vals.basis_values.size(); i++)
                        {
                            new_sol(global * dim + d) += vals.basis_values[i].val(0) * sol(bases[I(0)].bases[i].global()[0].index * dim + d);
                        }
                    }
                }
            }
            sol.swap(new_sol);
        }

        int dim;
        int n_el;
        int shape;

        RowVectorNd min_domain;
        RowVectorNd max_domain;

        Eigen::MatrixXd V;
        Eigen::MatrixXi T;
        igl::AABB<Eigen::MatrixXd, 2> tree;

        std::vector<std::list<int>> hash_table;
        int                         cell_num;
    };
}

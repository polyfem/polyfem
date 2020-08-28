#pragma once

#include <polyfem/Common.hpp>

#include <polysolve/FEMSolver.hpp>
#include <polyfem/Logger.hpp>

#include <polyfem/AssemblerUtils.hpp>
#include <memory>

#ifdef POLYFEM_WITH_TBB
#include <tbb/tbb.h>
#endif

using namespace polysolve;

namespace polyfem
{

    class OperatorSplittingSolver
    {
    public:
        void initialize_grid(const polyfem::Mesh& mesh, 
        const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases,
        const double& extra_resolution)
        {
            Eigen::MatrixXd p0, p1, p;
            mesh.get_edges(p0, p1);
            p = p0 - p1;
            resolution = p.rowwise().norm().minCoeff() / extra_resolution;

            grid_cell_num = RowVectorNd::Zero(dim);
            for(int d = 0; d < dim; d++)
            {
                grid_cell_num(d) = ceil((max_domain(d) - min_domain(d)) / resolution);
            }
            if(dim == 2)
                density = Eigen::VectorXd::Zero((grid_cell_num(0)+1) * (grid_cell_num(1)+1));
            else
                density = Eigen::VectorXd::Zero((grid_cell_num(0)+1) * (grid_cell_num(1)+1) * (grid_cell_num(2)+1));
        }

        void initialize_mesh(const polyfem::Mesh& mesh, 
        const int shape, const int n_el,
        const std::vector<polyfem::LocalBoundary>& local_boundary)
        {
            dim = mesh.dimension();
            mesh.bounding_box(min_domain, max_domain);

            const int size = local_boundary.size();
            boundary_elem_id.reserve(size);
            for(int e = 0; e < size; e++)
            {
                boundary_elem_id.push_back(local_boundary[e].element_id());
            }

            T.resize(n_el, shape);
            for (int e = 0; e < n_el; e++)
            {
                for (int i = 0; i < shape; i++)
                {
                    T(e, i) = mesh.cell_vertex_(e, i);
                }
            }
            V = Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);
            for (int i = 0; i < V.rows(); i++)
            {
                auto p = mesh.point(i);
                for (int d = 0; d < dim; d++)
                {
                    V(i, d) = p(d);
                }
                if (dim == 2) V(i, 2) = 0;
            }
        }

        void initialize_hashtable()
        {
            hash_table_cell_num = (int)pow(n_el, 1./dim);
            hash_table.resize((int)pow(hash_table_cell_num, dim));
            for(int e = 0; e < T.rows(); e++)
            {
                Eigen::VectorXd min_ = V.row(T(e, 0));
                Eigen::VectorXd max_ = min_;

                for(int i = 1; i < T.cols(); i++)
                {
                    Eigen::VectorXd p = V.row(T(e, i));
                    for(int d = 0; d < dim; d++)
                    {
                        if(min_(d) > p(d)) min_(d) = p(d);
                        if(max_(d) < p(d)) max_(d) = p(d);
                    }
                }

                Eigen::VectorXi min_int(dim), max_int(dim);

                for(int d = 0; d < dim; d++)
                {
                    double temp = hash_table_cell_num / (max_domain(d) - min_domain(d));
                    min_int(d) = floor((min_(d) - min_domain(d)) * temp);
                    max_int(d) = ceil((max_(d) - min_domain(d)) * temp);

                    if(min_int(d) < 0) 
                        min_int(d) = 0;
                    if(max_int(d) > hash_table_cell_num)
                        max_int(d) = hash_table_cell_num;
                }

                for(int x = min_int(0); x < max_int(0); x++)
                {
                    for(int y = min_int(1); y < max_int(1); y++)
                    {
                        if(dim == 2)
                        {
                            int idx = x + y * hash_table_cell_num;
                            hash_table[idx].push_front(e);
                        }
                        else
                        {
                            for(int z = min_int(2); z < max_int(2); z++)
                            {
                                int idx = x + (y + z * hash_table_cell_num) * hash_table_cell_num;
                                hash_table[idx].push_front(e);
                            }
                        }
                    }
                }
            }
        }

        OperatorSplittingSolver() {}

        void initialize_solver(const polyfem::Mesh& mesh,
        const int shape_, const int n_el_, 
        const std::vector<polyfem::LocalBoundary>& local_boundary,
        const std::vector<int>& boundary_nodes_)
        {
            shape = shape_;
            n_el = n_el_;
            boundary_nodes = boundary_nodes_;

            initialize_mesh(mesh, shape, n_el, local_boundary);
            initialize_hashtable();
        }

        OperatorSplittingSolver(const polyfem::Mesh& mesh,
        const int shape, const int n_el, 
        const std::vector<polyfem::LocalBoundary>& local_boundary,
        const std::vector<int>& boundary_nodes)
        {
            initialize_solver(mesh, shape, n_el, local_boundary, boundary_nodes);
        }

        OperatorSplittingSolver(const polyfem::Mesh& mesh,
        const int shape, const int n_el, 
        const std::vector<polyfem::LocalBoundary>& local_boundary,
        const std::vector<int>& boundary_nodes,
        const StiffnessMatrix& mass,
        const StiffnessMatrix& stiffness_viscosity,
        const StiffnessMatrix& stiffness_velocity,
        const double& dt,
        const double& viscosity_,
        const std::string &solver_type, 
        const std::string &precond,
        const json& params,
        const std::string &save_path) : solver_type(solver_type), precond(precond), params(params)
        {
            initialize_solver(mesh, shape, n_el, local_boundary, boundary_nodes);

            mat_diffusion = mass + viscosity_ * dt * stiffness_viscosity;
            
            solver_diffusion = LinearSolver::create(solver_type, precond);
            solver_diffusion->setParameters(params);
            logger().info("{}...", solver_diffusion->name());
            if (solver_type == "Pardiso" || solver_type == "Eigen::SimplicialLDLT" || solver_type == "Eigen::SparseLU")
            {
                StiffnessMatrix mat1 = mat_diffusion;
                prefactorize(*solver_diffusion, mat1, boundary_nodes, mat1.rows(), save_path);
            }

            mat_projection.resize(stiffness_velocity.rows() + 1, stiffness_velocity.cols() + 1);

            std::vector<Eigen::Triplet<double> > coefficients;
            coefficients.reserve(stiffness_velocity.nonZeros() + 2 * stiffness_velocity.rows());

            for(int i = 0; i < stiffness_velocity.outerSize(); i++)
            {
                for(StiffnessMatrix::InnerIterator it(stiffness_velocity,i); it; ++it)
                {
                    coefficients.emplace_back(it.row(),it.col(),it.value());
                }
            }

            const double val = 1. / (mat_projection.rows() - 1);
            for (int i = 0; i < mat_projection.rows() - 1; i++)
            {
                coefficients.emplace_back(i, mat_projection.cols() - 1, val);
                coefficients.emplace_back(mat_projection.rows() - 1, i, val);
            }

            mat_projection.setFromTriplets(coefficients.begin(), coefficients.end());
            solver_projection = LinearSolver::create(solver_type, precond);
            solver_projection->setParameters(params);
            logger().info("{}...", solver_projection->name());
            if (solver_type == "Pardiso" || solver_type == "Eigen::SimplicialLDLT" || solver_type == "Eigen::SparseLU")
            {
                StiffnessMatrix mat2 = mat_projection;
                prefactorize(*solver_projection, mat2, std::vector<int>(), mat2.rows() - 1, save_path);
            }
        }

        void export_3d_mesh(const std::string& export_mesh_path)
        {
            std::ofstream FILE(export_mesh_path.c_str(), std::ios::out);
            const int n_v = V.rows();
            const int n_e = T.rows();

            FILE << "MeshVersionFormatted 1\nDimension 3\n";
            FILE << "Vertices\n" << n_v << std::endl;
            for(int i = 0; i < n_v; i++)
                FILE << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << " 0\n";
            if(T.cols() == 4)
                FILE << "Tetrahedra\n" << n_e << std::endl;
            else
                FILE << "Hexahedra\n" << n_e << std::endl;
            for(int i = 0; i < n_e; i++)
            {
                for(int j = 0; j < T.cols(); j++)
                    FILE << T(i, j)+1 << " ";
                FILE << "0\n";
            }
            FILE << "End\n";
            FILE.close();
        }

        int handle_boundary_advection(RowVectorNd& pos)
        {
            double dist = 1e10;
            int idx = -1, local_idx = -1;
            const int size = boundary_elem_id.size();
#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, size, 1, [&](int e)
#else
            for(int e = 0; e < size; e++)
#endif
            {
                int elem_idx = boundary_elem_id[e];

                for (int i = 0; i < shape; i++)
                {
                    double dist_ = 0;
                    for(int d = 0; d < dim; d++)
                    {
                        dist_ += pow( pos(d) - V(T(elem_idx, i), d), 2);
                    }
                    dist_ = sqrt(dist_);
                    if(dist_ < dist)
                    {
                        dist = dist_;
                        idx = elem_idx;
                        local_idx = i;
                    }
                }
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif
            for(int d = 0; d < dim; d++)
                pos(d) = V(T(idx, local_idx), d);
            return idx;
        }

        void trace_back(const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        const RowVectorNd& pos_1, 
        const RowVectorNd& vel_1, 
        RowVectorNd& pos_2, 
        RowVectorNd& vel_2, 
        const Eigen::MatrixXd& sol,
        const double dt)
        {
            int new_elem;
            Eigen::MatrixXd local_pos;

            pos_2 = pos_1 - vel_1 * dt;

            if((new_elem = search_cell(gbases, pos_2, local_pos)) == -1)
            {
                new_elem = handle_boundary_advection(pos_2);
                calculate_local_pts(gbases[new_elem], new_elem, pos_2, local_pos);
            }

            // interpolation
            vel_2 = RowVectorNd::Zero(dim);
            ElementAssemblyValues vals;
            vals.compute(new_elem, dim == 3, local_pos, bases[new_elem], gbases[new_elem]);
            for (int d = 0; d < dim; d++)
            {
                for (int i = 0; i < vals.basis_values.size(); i++)
                {
                    vel_2(d) += vals.basis_values[i].val(0) * sol(bases[new_elem].bases[i].global()[0].index * dim + d);
                }
            }
        }

        void trace_back_density(const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        const RowVectorNd& pos_1,
        double& val, 
        const Eigen::MatrixXd& sol,
        const double dt)
        {
            val = 0;
            int elem_id;
            Eigen::MatrixXd local_pos;
            elem_id = search_cell(gbases, pos_1, local_pos);
            if(elem_id == -1) return; // density outside of domain = 0

            // vel_2 <- velocity at pos_1
            calculate_local_pts(gbases[elem_id], elem_id, pos_1, local_pos);
            RowVectorNd vel_2 = RowVectorNd::Zero(dim);
            ElementAssemblyValues vals;
            vals.compute(elem_id, dim == 3, local_pos, bases[elem_id], gbases[elem_id]);
            for (int d = 0; d < dim; d++)
            {
                for (int i = 0; i < vals.basis_values.size(); i++)
                {
                    vel_2(d) += vals.basis_values[i].val(0) * sol(bases[elem_id].bases[i].global()[0].index * dim + d);
                }
            }
            // RK1
            RowVectorNd pos_2 = pos_1 - vel_2 * dt;

            Eigen::VectorXi int_pos(dim);
            Eigen::MatrixXd weights(2, dim);
            for(int d = 0; d < dim; d++)
            {
                int_pos(d) = floor((pos_2(d) - min_domain(d)) / resolution);
                if(int_pos(d) < 0 || int_pos(d) >= grid_cell_num(d)) return;
                weights(1, d) = (pos_2(d) - min_domain(d)) / resolution - int_pos(d);
                weights(0, d) = 1 - weights(1, d);
            }
            
            for(int d1 = 0; d1 < 2; d1++)
            {
                for(int d2 = 0; d2 < 2; d2++)
                {
                    if(dim == 2)
                    {
                        const int idx = (int_pos(0) + d1) + (int_pos(1) + d2) * (grid_cell_num(0)+1);
                        val += density(idx) * weights(d1, 0) * weights(d2, 1);
                    }
                    else
                    {
                        for(int d3 = 0; d3 < 2; d3++)
                        {
                            const int idx = (int_pos(0) + d1) + (int_pos(1) + d2 + (int_pos(2) + d3) * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
                            val += density(idx) * weights(d1, 0) * weights(d2, 1) * weights(d3, 2);
                        }
                    }
                }
            }
        }

        void advection(const polyfem::Mesh& mesh, 
        const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        Eigen::MatrixXd& sol, 
        const double dt, 
        const Eigen::MatrixXd& local_pts, 
        const int order = 1,
        const int RK = 1)
        {
            // to store new velocity
            Eigen::MatrixXd new_sol = Eigen::MatrixXd::Zero(sol.size(), 1);
            // number of FEM nodes
            const int n_vert = sol.size() / dim;
            Eigen::VectorXi traversed = Eigen::VectorXi::Zero(n_vert);

#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, n_el, 1, [&](int e)
#else
            for (int e = 0; e < n_el; ++e)
#endif
            {
                // geometry vertices of element e
                std::vector<RowVectorNd> vert(shape, RowVectorNd::Zero(1, dim));
                for (int i = 0; i < shape; i++)
                {
                    int tmp = mesh.cell_vertex_(e, i);
                    for(int d = 0; d < dim; d++)
                        vert[i](d) = mesh.point(tmp)(d);
                }

                // to compute global position with barycentric coordinate
                ElementAssemblyValues gvals;
                gvals.compute(e, dim == 3, local_pts, gbases[e], gbases[e]);

                for (int i = 0; i < local_pts.rows(); i++)
                {
                    // global index of this FEM node
                    int global = bases[e].bases[i].global()[0].index;

                    if (traversed(global)) continue;
                    traversed(global) = 1;

                    RowVectorNd vel_1[4], pos_1[4];

                    // velocity of this FEM node
                    vel_1[0] = sol.block(global * dim, 0, dim, 1).transpose();

                    // global position of this FEM node
                    pos_1[0] = RowVectorNd::Zero(1, dim);
                    for (int j = 0; j < shape; j++)
                    {
                        pos_1[0] += gvals.basis_values[j].val(i) * vert[j];
                    }

                    if(RK>=3)
                    {
                        trace_back( gbases, bases, pos_1[0], vel_1[0], pos_1[1], vel_1[1], sol, 0.5 * dt);
                        trace_back( gbases, bases, pos_1[0], vel_1[1], pos_1[2], vel_1[2], sol, 0.75 * dt);
                        trace_back( gbases, bases, pos_1[0], 2 * vel_1[0] + 3 * vel_1[1] + 4 * vel_1[2], pos_1[3], vel_1[3], sol, dt / 9);
                    }
                    else if(RK==2)
                    {
                        trace_back( gbases, bases, pos_1[0], vel_1[0], pos_1[1], vel_1[1], sol, 0.5 * dt);
                        trace_back( gbases, bases, pos_1[0], vel_1[1], pos_1[3], vel_1[3], sol, dt);
                    }
                    else if(RK==1)
                    {
                        trace_back( gbases, bases, pos_1[0], vel_1[0], pos_1[3], vel_1[3], sol, dt);
                    }

                    new_sol.block(global * dim, 0, dim, 1) = vel_1[3].transpose();

                    if(order == 2)
                    {
                        RowVectorNd vel_2[3], pos_2[3];

                        if(RK>=3)
                        {
                            trace_back( gbases, bases, pos_1[3], vel_1[3], pos_2[0], vel_2[0], sol, -0.5 * dt);
                            trace_back( gbases, bases, pos_1[3], vel_1[1], pos_2[1], vel_2[1], sol, -0.75 * dt);
                            trace_back( gbases, bases, pos_1[3], 2 * vel_1[3] + 3 * vel_2[0] + 4 * vel_2[1], pos_2[2], vel_2[2], sol, -dt / 9);
                        }
                        else if(RK==2)
                        {
                            trace_back( gbases, bases, pos_1[3], vel_1[3], pos_2[0], vel_2[0], sol, -0.5 * dt);
                            trace_back( gbases, bases, pos_1[3], vel_2[0], pos_2[2], vel_2[2], sol, -dt);
                        }
                        else if(RK==1)
                        {
                            trace_back( gbases, bases, pos_1[3], vel_1[3], pos_2[2], vel_2[2], sol, -dt);
                        }
                        
                        new_sol.block(global * dim, 0, dim, 1) += (vel_1[0] - vel_2[2]).transpose() / 2;
                    }
                }
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif
            sol.swap(new_sol);
        }

        void advect_density(const std::vector<polyfem::ElementBases>& gbases,
        const std::vector<polyfem::ElementBases>& bases, 
        const Eigen::MatrixXd& sol, 
        const double dt)
        {
            Eigen::VectorXd new_density = Eigen::VectorXd::Zero(density.size());
            RowVectorNd pos(1, dim);
            for(int i = 0; i <= grid_cell_num(0); i++)
            {
                pos(0) = i * resolution + min_domain(0);
                for(int j = 0; j <= grid_cell_num(1); j++)
                {
                    pos(1) = j * resolution + min_domain(1);
                    if(dim == 2)
                    {
                        const int idx = i + j * (grid_cell_num(0)+1);
                        trace_back_density(gbases, bases, pos, new_density[idx], sol, dt);
                    }
                    else
                    {
                        for(int k = 0; k <= grid_cell_num(2); k++)
                        {
                            pos(2) = k * resolution + min_domain(2);
                            const int idx = i + (j + k * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
                            trace_back_density(gbases, bases, pos, new_density[idx], sol, dt);
                        }
                    }
                }
            }
            density.swap(new_density);
        }
        
        void advection_FLIP(const polyfem::Mesh& mesh, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, Eigen::MatrixXd& sol, const double dt, const Eigen::MatrixXd& local_pts, const int order = 1)
        {
            const int ppe = shape; // particle per element
            const double FLIPRatio = 1;
            // initialize or resample particles and update velocity via g2p
            if (position_particle.empty()) {
                // initialize particles
                position_particle.resize(n_el * ppe);
                velocity_particle.resize(n_el * ppe);
                cellI_particle.resize(n_el * ppe);
#ifdef POLYFEM_WITH_TBB
                tbb::parallel_for(0, n_el, 1, [&](int e)
#else
                for (int e = 0; e < n_el; ++e)
#endif
                {
                    // sample particle in element e
                    Eigen::MatrixXd local_pts_particle;
                    local_pts_particle.setRandom(ppe, dim);
                    local_pts_particle.array() += 1;
                    local_pts_particle.array() /= 2;

                    // geometry vertices of element e
                    std::vector<RowVectorNd> vert(shape);
                    for (int i = 0; i < shape; ++i)
                    {
                        cellI_particle[e * ppe + i] = e;
                        vert[i] = mesh.point(mesh.cell_vertex_(e, i));
                    }

                    // compute global position and velocity of particles
                    // construct interpolant (linear for position)
                    ElementAssemblyValues gvals;
                    gvals.compute(e, dim == 3, local_pts_particle, gbases[e], gbases[e]);
                    // construct interpolant (for velocity)
                    ElementAssemblyValues vals;
                    vals.compute(e, dim == 3, local_pts_particle, bases[e], gbases[e]); // possibly higher-order
                    for (int j = 0; j < ppe; ++j) {
                        position_particle[ppe * e + j].setZero(1, dim);
                        for (int i = 0; i < shape; ++i)
                        {
                            position_particle[ppe * e + j] += gvals.basis_values[i].val(j) * vert[i];
                        }
                        
                        velocity_particle[e * ppe + j].setZero(1, dim);
                        for (int i = 0; i < vals.basis_values.size(); ++i)
                        {
                            velocity_particle[e * ppe + j] += vals.basis_values[i].val(j) * 
                                sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1).transpose();
                        }
                    }
                }
#ifdef POLYFEM_WITH_TBB
                );
#endif
            }
            else {
                // resample particles
                // count particle per cell
                std::vector<int> counter(n_el, 0);
                std::vector<int> redundantPI;
                std::vector<bool> isRedundant(cellI_particle.size(), false);
                for (int pI = 0; pI < cellI_particle.size(); ++pI) {
                    ++counter[cellI_particle[pI]];
                    if (counter[cellI_particle[pI]] > ppe) {
                        redundantPI.emplace_back(pI);
                        isRedundant[pI] = true;
                    }
                }
                // g2p -- update velocity 
#ifdef POLYFEM_WITH_TBB
                tbb::parallel_for(0, (int)cellI_particle.size(), 1, [&](int pI)
#else
                for (int pI = 0; pI < cellI_particle.size(); ++pI) 
#endif
                {
                    if (!isRedundant[pI]) {
                        int e = cellI_particle[pI];
                        Eigen::MatrixXd local_pts_particle;
                        calculate_local_pts(gbases[e], e, position_particle[pI], local_pts_particle);
                        
                        ElementAssemblyValues vals;
                        vals.compute(e, dim == 3, local_pts_particle, bases[e], gbases[e]); // possibly higher-order
                        RowVectorNd FLIPdVel, PICVel;
                        FLIPdVel.setZero(1, dim);
                        PICVel.setZero(1, dim);
                        for (int i = 0; i < vals.basis_values.size(); ++i)
                        {
                            FLIPdVel += vals.basis_values[i].val(0) * 
                                (sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1) -
                                new_sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1)).transpose();
                            PICVel += vals.basis_values[i].val(0) * 
                                sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1).transpose();
                        }
                        velocity_particle[pI] = (1.0 - FLIPRatio) * PICVel + 
                            FLIPRatio * (velocity_particle[pI] + FLIPdVel);
                    }
                }
#ifdef POLYFEM_WITH_TBB
                );
#endif
                // resample
                for (int e = 0; e < n_el; ++e) {
                    if (counter[e] >= ppe) {
                        continue;
                    }

                    // geometry vertices of element e
                    std::vector<RowVectorNd> vert(shape);
                    for (int i = 0; i < shape; ++i)
                    {
                        vert[i] = mesh.point(mesh.cell_vertex_(e, i));
                    }
                    while (counter[e] < ppe) {
                        int pI = redundantPI.back();
                        redundantPI.pop_back();
                        
                        cellI_particle[pI] = e;

                        // sample particle in element e
                        Eigen::MatrixXd local_pts_particle;
                        local_pts_particle.setRandom(1, dim);
                        local_pts_particle.array() += 1;
                        local_pts_particle.array() /= 2;

                        // compute global position and velocity of particles
                        // construct interpolant (linear for position)
                        ElementAssemblyValues gvals;
                        gvals.compute(e, dim == 3, local_pts_particle, gbases[e], gbases[e]);
                        position_particle[pI].setZero(1, dim);
                        for (int i = 0; i < shape; ++i)
                        {
                            position_particle[pI] += gvals.basis_values[i].val(0) * vert[i];
                        }

                        // construct interpolant (for velocity)
                        ElementAssemblyValues vals;
                        vals.compute(e, dim == 3, local_pts_particle, bases[e], gbases[e]); // possibly higher-order
                        velocity_particle[pI].setZero(1, dim);
                        for (int i = 0; i < vals.basis_values.size(); ++i)
                        {
                            velocity_particle[pI] += vals.basis_values[i].val(0) * 
                                sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1).transpose();
                        }

                        ++counter[e];
                    }
                }
            }

            // advect
            std::vector<ElementAssemblyValues> velocity_interpolator(ppe * n_el);
#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, (int)(ppe * n_el), 1, [&](int pI)
#else
            for (int pI = 0; pI < ppe * n_el; ++pI) 
#endif
            {
                // update particle position via advection
                RowVectorNd newvel;
                trace_back( gbases, bases, position_particle[pI], velocity_particle[pI], 
                    position_particle[pI], newvel, sol, -dt);

                // RK3:
                // RowVectorNd bypass, vel2, vel3;
                // trace_back( gbases, bases, position_particle[pI], velocity_particle[pI], 
                //     bypass, vel2, sol, -0.5 * dt);
                // trace_back( gbases, bases, position_particle[pI], vel2, 
                //     bypass, vel3, sol, -0.75 * dt);
                // trace_back( gbases, bases, position_particle[pI], 
                //     2 * velocity_particle[pI] + 3 * vel2 + 4 * vel3, 
                //     position_particle[pI], bypass, sol, -dt / 9);

                // prepare P2G
                Eigen::VectorXi I(1);
                Eigen::MatrixXd local_pos;

                if((I(0) = search_cell(gbases, position_particle[pI], local_pos)) == -1)
                {
                    I(0) = handle_boundary_advection(position_particle[pI]);
                    calculate_local_pts(gbases[I(0)], I(0), position_particle[pI], local_pos);
                }

                // construct interpolator (always linear for P2G, can use gaussian or bspline later)
                velocity_interpolator[pI].compute(I(0), dim == 3, local_pos, gbases[I(0)], gbases[I(0)]);
                cellI_particle[pI] = I(0);
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif

            // P2G
            new_sol = Eigen::MatrixXd::Zero(sol.size(), 1);
            new_sol_w = Eigen::MatrixXd::Zero(sol.size() / dim, 1);
            new_sol_w.array() += 1e-13;
            for (int pI = 0; pI < ppe * n_el; ++pI) {
                int cellI = cellI_particle[pI];
                ElementAssemblyValues& vals = velocity_interpolator[pI];
                for (int i = 0; i < vals.basis_values.size(); ++i)
                {
                    new_sol.block(bases[cellI].bases[i].global()[0].index * dim, 0, dim, 1) += 
                        vals.basis_values[i].val(0) * velocity_particle[pI].transpose();
                    new_sol_w(bases[cellI].bases[i].global()[0].index) += vals.basis_values[i].val(0);
                }
            }
            //TODO: need to add up boundary velocities and weights because of perodic BC

#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, (int)new_sol.rows() / dim, 1, [&](int i)
#else
            for (int i = 0; i < new_sol.rows() / dim; ++i) 
#endif
            {
                new_sol.block(i * dim, 0, dim, 1) /= new_sol_w(i, 0);
                sol.block(i * dim, 0, dim, 1) = new_sol.block(i * dim, 0, dim, 1);
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif
        }

        void advection_PIC(const polyfem::Mesh& mesh, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, Eigen::MatrixXd& sol, const double dt, const Eigen::MatrixXd& local_pts, const int order = 1)
        {
            // to store new velocity and weights for particle grid transfer
            Eigen::MatrixXd new_sol = Eigen::MatrixXd::Zero(sol.size(), 1);
            Eigen::MatrixXd new_sol_w = Eigen::MatrixXd::Zero(sol.size() / dim, 1);
            new_sol_w.array() += 1e-13;

            const int ppe = shape; // particle per element
            std::vector<ElementAssemblyValues> velocity_interpolator(ppe * n_el);
            position_particle.resize(ppe * n_el);
            velocity_particle.resize(ppe * n_el);
            cellI_particle.resize(ppe * n_el);
#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, n_el, 1, [&](int e)
#else
            for (int e = 0; e < n_el; ++e)
#endif
            {
                // resample particle in element e
                Eigen::MatrixXd local_pts_particle;
                local_pts_particle.setRandom(ppe, dim);
                local_pts_particle.array() += 1;
                local_pts_particle.array() /= 2;

                // geometry vertices of element e
                std::vector<RowVectorNd> vert(shape);
                for (int i = 0; i < shape; ++i)
                {
                    vert[i] = mesh.point(mesh.cell_vertex_(e, i));
                }

                // construct interpolant (linear for position)
                ElementAssemblyValues gvals;
                gvals.compute(e, dim == 3, local_pts_particle, gbases[e], gbases[e]);

                // compute global position of particles
                for (int i = 0; i < ppe; ++i)
                {
                    position_particle[ppe * e + i].setZero(1, dim);
                    for (int j = 0; j < shape; ++j)
                    {
                        position_particle[ppe * e + i] += gvals.basis_values[j].val(i) * vert[j];
                    }
                }

                // compute velocity
                ElementAssemblyValues vals;
                vals.compute(e, dim == 3, local_pts_particle, bases[e], gbases[e]); // possibly higher-order
                for (int j = 0; j < ppe; ++j) {
                    velocity_particle[e * ppe + j].setZero(1, dim);
                    for (int i = 0; i < vals.basis_values.size(); ++i)
                    {
                        velocity_particle[e * ppe + j] += vals.basis_values[i].val(j) * 
                            sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1).transpose();
                    }
                }

                // update particle position via advection
                for (int i = 0; i < ppe; ++i) {
                    RowVectorNd newvel;
                    trace_back( gbases, bases, position_particle[ppe * e + i], velocity_particle[e * ppe + i], 
                        position_particle[ppe * e + i], newvel, sol, -dt);

                    // RK3:
                    // RowVectorNd bypass, vel2, vel3;
                    // trace_back( gbases, bases, position_particle[ppe * e + i], velocity_particle[e * ppe + i], 
                    //     bypass, vel2, sol, -0.5 * dt);
                    // trace_back( gbases, bases, position_particle[ppe * e + i], vel2, 
                    //     bypass, vel3, sol, -0.75 * dt);
                    // trace_back( gbases, bases, position_particle[ppe * e + i], 
                    //     2 * velocity_particle[e * ppe + i] + 3 * vel2 + 4 * vel3, 
                    //     position_particle[ppe * e + i], bypass, sol, -dt / 9);
                }

                // prepare P2G
                for (int j = 0; j < ppe; ++j) {
                    Eigen::VectorXi I(1);
                    Eigen::MatrixXd local_pos;
                    
                    // find cell
                    if((I(0) = search_cell(gbases, position_particle[ppe * e + j],local_pos)) == -1)
                    {
                        I(0) = handle_boundary_advection(position_particle[ppe * e + j]);
                        calculate_local_pts(gbases[I(0)], I(0), position_particle[ppe * e + j], local_pos);
                    }

                    // construct interpolator (always linear for P2G, can use gaussian or bspline later)
                    velocity_interpolator[ppe * e + j].compute(I(0), dim == 3, local_pos, gbases[I(0)], gbases[I(0)]);
                    cellI_particle[ppe * e + j] = I(0);
                }
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif

            // P2G
            for (int e = 0; e < n_el; ++e)
            {
                for (int j = 0; j < ppe; ++j) 
                {
                    int cellI = cellI_particle[ppe * e + j];
                    ElementAssemblyValues& vals = velocity_interpolator[ppe * e + j];
                    for (int i = 0; i < vals.basis_values.size(); ++i)
                    {
                        new_sol.block(bases[cellI].bases[i].global()[0].index * dim, 0, dim, 1) += 
                            vals.basis_values[i].val(0) * velocity_particle[ppe * e + j].transpose();
                        new_sol_w(bases[cellI].bases[i].global()[0].index) += vals.basis_values[i].val(0);
                    }
                }
            }
            //TODO: need to add up boundary velocities and weights because of perodic BC

#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, (int)new_sol.rows() / dim, 1, [&](int i)
#else
            for (int i = 0; i < new_sol.rows() / dim; ++i) 
#endif
            {
                sol.block(i * dim, 0, dim, 1) = new_sol.block(i * dim, 0, dim, 1) / new_sol_w(i, 0);
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif
            //TODO: need to think about what to do with negative quadratic weight
        }

        void set_bc(const polyfem::Mesh& mesh, 
        const std::vector<int>& bnd_nodes,
        const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        Eigen::MatrixXd& sol, 
        const Eigen::MatrixXd& local_pts, 
        const std::shared_ptr<Problem> problem, 
        const double time)
        {
            const int size = boundary_elem_id.size();
#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, size, 1, [&](int e)
#else
            for(int e = 0; e < size; e++)
#endif
            {
                int elem_idx = boundary_elem_id[e];

                // geometry vertices of element e
                Eigen::MatrixXd vert(shape, dim);
                for (int i = 0; i < shape; i++)
                {
                    for(int d = 0; d < dim; d++)
                        vert(i, d) = V(T(elem_idx, i), d);
                }

                ElementAssemblyValues gvals;
                gvals.compute(elem_idx, dim == 3, local_pts, gbases[elem_idx], gbases[elem_idx]);

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
                            pos(0, d) += gvals.basis_values[j].val(local_idx) * vert(j, d);
                        }
                    }

                    Eigen::MatrixXd val;
                    problem->bc(mesh, Eigen::MatrixXi::Zero(1,1), Eigen::MatrixXd::Zero(1,1), pos, time, val);

                    for (int d = 0; d < dim; d++)
                    {
                        sol(global_idx * dim + d) = val(d);
                    }
                }
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif
        }

        void solve_diffusion_1st(const StiffnessMatrix& mass, const std::vector<int>& bnd_nodes, Eigen::MatrixXd& sol)
        {
            Eigen::VectorXd rhs;
            
            for(int d = 0; d < dim; d++)
            {
                Eigen::VectorXd x(sol.size() / dim);
                for(int j = 0; j < x.size(); j++)
                {
                    x(j) = sol(j * dim + d);
                }
                rhs = mass * x;

                // keep dirichlet bc
                for (int i = 0; i < bnd_nodes.size(); i++)
                {
                    rhs(bnd_nodes[i]) = x(bnd_nodes[i]);
                }

                if (solver_type == "Pardiso" || solver_type == "Eigen::SimplicialLDLT" || solver_type == "Eigen::SparseLU")
                {
                    dirichlet_solve_prefactorized(*solver_diffusion, mat_diffusion, rhs, bnd_nodes, x);
                }
                else
                {
                    dirichlet_solve(*solver_diffusion, mat_diffusion, rhs, bnd_nodes, x, mat_diffusion.rows(), "", false);
                }
                

                for(int j = 0; j < x.size(); j++)
                {
                    sol(j * dim + d) = x(j);
                }
            }
        }

        void external_force(const polyfem::Mesh& mesh,
        const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        const double dt, 
        Eigen::MatrixXd& sol, 
        const Eigen::MatrixXd& local_pts, 
        const std::shared_ptr<Problem> problem, 
        const double time)
        {
#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, n_el, 1, [&](int e)
#else
            for(int e = 0; e < n_el; e++)
#endif
            {
                ElementAssemblyValues gvals;
                gvals.compute(e, dim == 3, local_pts, gbases[e], gbases[e]);

                for (int local_idx = 0; local_idx < bases[e].bases.size(); local_idx++)
                {
                    int global_idx = bases[e].bases[local_idx].global()[0].index;

                    Eigen::MatrixXd pos = Eigen::MatrixXd::Zero(1, dim);
                    for (int j = 0; j < shape; j++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            pos(0, d) += gvals.basis_values[j].val(local_idx) * V(T(e, j), d);
                        }
                    }

                    Eigen::MatrixXd val;
                    problem->rhs(std::string(), pos, time, val);

                    for (int d = 0; d < dim; d++)
                    {
                        sol(global_idx * dim + d) += val(d) * dt;
                    }
                }
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif
        }

        void solve_pressure(const StiffnessMatrix& mixed_stiffness, Eigen::MatrixXd& sol, Eigen::MatrixXd& pressure)
        {
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(mixed_stiffness.rows() + 1); // mixed_stiffness * sol;
            Eigen::VectorXd temp = mixed_stiffness * sol;
            for(int i = 0; i < temp.rows(); i++)
            {
                rhs(i) = temp(i);
            }

            Eigen::VectorXd x(pressure.size());
            for(int i = 0; i < pressure.size(); i++)
            {
                x(i) = pressure(i);
            }
            if (solver_type == "Pardiso" || solver_type == "Eigen::SimplicialLDLT" || solver_type == "Eigen::SparseLU")
            {
                dirichlet_solve_prefactorized(*solver_projection, mat_projection, rhs, std::vector<int>(), x);
            }
            else
            {
                dirichlet_solve(*solver_projection, mat_projection, rhs, std::vector<int>(), x, mat_projection.rows() - 1, "", false);
            }
            
            pressure = x;
        }

        void projection(const polyfem::Mesh& mesh, 
        int n_bases, 
        const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        const std::vector<polyfem::ElementBases>& pressure_bases, 
        const Eigen::MatrixXd& local_pts, 
        Eigen::MatrixXd& pressure, 
        Eigen::MatrixXd& sol)
        {
            Eigen::VectorXd grad_pressure = Eigen::VectorXd::Zero(n_bases * dim);
            Eigen::VectorXi traversed = Eigen::VectorXi::Zero(n_bases);

            ElementAssemblyValues vals;
            for (int e = 0; e < n_el; ++e)
            {
                vals.compute(e, dim == 3, local_pts, pressure_bases[e], gbases[e]);
                for (int j = 0; j < local_pts.rows(); j++)
                {
                    int global_ = bases[e].bases[j].global()[0].index;
                    for (int i = 0; i < vals.basis_values.size(); i++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            assert(pressure_bases[e].bases[i].global().size() == 1);
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

        void initialize_solution(const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        const std::shared_ptr<Problem> problem, 
        Eigen::MatrixXd& sol, 
        const Eigen::MatrixXd& local_pts)
        {
#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, n_el, 1, [&](int e)
#else
            for (int e = 0; e < n_el; ++e)
#endif
            {
                // to compute global position with barycentric coordinate
                ElementAssemblyValues gvals;
                gvals.compute(e, dim == 3, local_pts, gbases[e], gbases[e]);

                for (int i = 0; i < local_pts.rows(); i++)
                {
                    Eigen::MatrixXd pts = Eigen::MatrixXd::Zero(1, dim);
                    for (int j = 0; j < shape; j++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            pts(0, d) += V(T(e, j), d) * gvals.basis_values[j].val(i);
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
#ifdef POLYFEM_WITH_TBB
            );
#endif
        }

        void initialize_density(const std::shared_ptr<Problem> problem)
        {
            Eigen::MatrixXd pts(1, dim);
            Eigen::MatrixXd tmp;
            for(int i = 0; i <= grid_cell_num(0); i++)
            {
                pts(0, 0) = i * resolution + min_domain(0);
                for(int j = 0; j <= grid_cell_num(1); j++)
                {
                    pts(0, 1) = j * resolution + min_domain(1);
                    if(dim == 2)
                    {
                        const int idx = i + j * (grid_cell_num(0)+1);
                        problem->initial_density(pts, tmp);
                        density(idx) = tmp(0);
                    }
                    else
                    {
                        for(int k = 0; k <= grid_cell_num(2); k++)
                        {
                            pts(0, 2) = k * resolution + min_domain(2);
                            const int idx = i + (j + k * (grid_cell_num(1)+1)) * (grid_cell_num(0)+1);
                            problem->initial_density(pts, tmp);
                            density(idx) = tmp(0);
                        }
                    }
                }
            }
        }

        int search_cell(const std::vector<polyfem::ElementBases>& gbases, const RowVectorNd& pos, Eigen::MatrixXd& local_pts)
        {
            Eigen::VectorXi pos_int(dim);
            for(int d = 0; d < dim; d++)
            {
                pos_int(d) = floor((pos(d) - min_domain(d)) / (max_domain(d) - min_domain(d)) * hash_table_cell_num);
                if(pos_int(d) < 0) pos_int(d) = 0;
                else if(pos_int(d) >= hash_table_cell_num) pos_int(d) = hash_table_cell_num - 1;
            }

            int idx = 0, dim_num = 1;
            for(int d = 0; d < dim; d++)
            {
                idx += pos_int(d) * dim_num;
                dim_num *= hash_table_cell_num;
            }

            const std::list<int>& list = hash_table[idx];
            for(auto it = list.begin(); it != list.end(); it++)
            {
                calculate_local_pts(gbases[*it], *it, pos, local_pts);

                if(shape == dim + 1)
                {
                    if(local_pts.minCoeff() > -1e-13 && local_pts.sum() < 1 + 1e-13)
                        return *it;
                }
                else
                {
                    if(local_pts.minCoeff() > -1e-13 && local_pts.maxCoeff() < 1 + 1e-13)
                        return *it;
                }
            }
            return -1; // not inside any elem
        }

        bool outside_quad(const std::vector<RowVectorNd>& vert, const RowVectorNd& pos)
        {
            double a = (vert[1](0) - vert[0](0)) * (pos(1) - vert[0](1)) - (vert[1](1)-vert[0](1)) * (pos(0) - vert[0](0));
            double b = (vert[2](0) - vert[1](0)) * (pos(1) - vert[1](1)) - (vert[2](1)-vert[1](1)) * (pos(0) - vert[1](0));
            double c = (vert[3](0) - vert[2](0)) * (pos(1) - vert[2](1)) - (vert[3](1)-vert[2](1)) * (pos(0) - vert[2](0));
            double d = (vert[0](0) - vert[3](0)) * (pos(1) - vert[3](1)) - (vert[0](1)-vert[3](1)) * (pos(0) - vert[3](0));

            if((a > 0 && b > 0 && c > 0 && d > 0) || (a < 0 && b < 0 && c < 0 && d < 0))
                return false;
            return true;
        }

        void calculate_local_pts(const polyfem::ElementBases& gbase, 
        const int elem_idx,
        const RowVectorNd& pos, 
        Eigen::MatrixXd& local_pos)
        {
            local_pos = Eigen::MatrixXd::Zero(1, dim);
            
            std::vector<RowVectorNd> vert(shape,RowVectorNd::Zero(1, dim));
            for (int i = 0; i < shape; i++)
            {
                for(int d = 0; d < dim; d++)
                    vert[i](d) = V(T(elem_idx, i), d);
            }
            // if(shape == 4 && dim == 2 && outside_quad(vert, pos))
            // {
            //     local_pos(0) = local_pos(1) = -1;
            //     return;
            // }
            Eigen::MatrixXd res;
            int iter_times = 0;
            int max_iter = 20;
            do
            {
                res = -pos;
                ElementAssemblyValues gvals_;
                gvals_.compute(elem_idx, dim == 3, local_pos, gbase, gbase);
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
                iter_times++;
            }
            while(res.norm() > 1e-12 && iter_times < max_iter);

            if(iter_times >= max_iter)
            {
                for(int d=0; d<dim; d++)
                    local_pos(d) = -1;
            }
        }

        void save_density();
        
        int dim;
        int n_el;
        int shape;

        RowVectorNd min_domain;
        RowVectorNd max_domain;

        Eigen::MatrixXd V;
        Eigen::MatrixXi T;

        std::vector<std::list<int>> hash_table;
        int                         hash_table_cell_num;

        std::vector<RowVectorNd> position_particle;
		std::vector<RowVectorNd> velocity_particle;
        std::vector<int> cellI_particle;
        Eigen::MatrixXd new_sol;
        Eigen::MatrixXd new_sol_w;

        std::vector<int> boundary_elem_id;
        std::vector<int> boundary_nodes;

        std::unique_ptr<polysolve::LinearSolver> solver_diffusion;
        std::unique_ptr<polysolve::LinearSolver> solver_projection;

        StiffnessMatrix mat_diffusion;
        StiffnessMatrix mat_projection;

        std::string solver_type;
        std::string precond;
        json params;

        Eigen::VectorXd density;
        // Eigen::VectorXi density_cell_no;
        // std::vector<ElementAssemblyValues> density_local_weights;
        RowVectorNd grid_cell_num;
        double resolution;
    };
}

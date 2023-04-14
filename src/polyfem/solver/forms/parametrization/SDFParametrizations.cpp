#include "SDFParametrizations.hpp"

#include <polyfem/utils/IsosurfaceInflator.hpp>
#include <polyfem/io/MshReader.hpp>
#include <polyfem/utils/Timer.hpp>
#include <igl/writeMSH.h>
#include <polyfem/io/VTUWriter.hpp>

#include <polyfem/State.hpp>
#include <polysolve/FEMSolver.hpp>

namespace polyfem::solver
{
    namespace {
        void write_msh(const std::string &path, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
        {
            Eigen::MatrixXd Vsave;
            Vsave.setZero(V.rows(), 3);
            Vsave.leftCols(V.cols()) = V;

            const int dim = V.cols();
            Eigen::MatrixXi Tri = (dim == 3) ? Eigen::MatrixXi() : F;
            Eigen::MatrixXi Tet = (dim == 3) ? F : Eigen::MatrixXi();

            igl::writeMSH(path, Vsave, Tri, Tet, Eigen::MatrixXi::Zero(Tri.rows(), 1), Eigen::MatrixXi::Zero(Tet.rows(), 1), std::vector<std::string>(), std::vector<Eigen::MatrixXd>(), std::vector<std::string>(), std::vector<Eigen::MatrixXd>(), std::vector<Eigen::MatrixXd>());
        }

        template <typename T>
        std::string to_string_with_precision(const T a_value, const int n = 6)
        {
            std::ostringstream out;
            out.precision(n);
            out << std::fixed << a_value;
            return out.str();
        }
    }

    bool SDF2Mesh::isosurface_inflator(const Eigen::VectorXd &x) const
    {
        if (last_x.size() == x.size() && last_x == x)
            return true;

        // if (false)
        // {
        //     std::string sdf_velocity_path_ = "tmp-vel.msh";
        //     std::string shape_params = "--params \"";
        //     for (int i = 0; i < x.size(); i++)
        //         shape_params += to_string_with_precision(x(i), 16) + " ";
        //     shape_params += "\" ";

        //     std::string command = "~/microstructures/build/isosurface_inflator/isosurface_cli 2D_doubly_periodic " + wire_path_ + " " + shape_params + " -S " + sdf_velocity_path_ + " " + out_path_;

        //     int return_val;
        //     try 
        //     {
        //         return_val = system(command.c_str());
        //     }
        //     catch (const std::exception &err)
        //     {
        //         logger().error("remesh command \"{}\" returns {}", command, return_val);

        //         return false;
        //     }

        //     logger().info("remesh command \"{}\" returns {}", command, return_val);

        //     {
        //         std::vector<std::vector<int>> elements;
        //         std::vector<std::vector<double>> weights;
        //         std::vector<int> body_ids;
        //         std::vector<std::string> node_data_name;
        //         std::vector<std::vector<double>> node_data;
        //         io::MshReader::load(sdf_velocity_path_, Vout, Fout, elements, weights, body_ids, node_data_name, node_data);

        //         assert(node_data_name.size() == node_data.size());

        //         vertex_normals = Eigen::Map<Eigen::MatrixXd>(node_data[0].data(), 3, Vout.rows()).topRows(Vout.cols()).transpose();
                
        //         shape_vel.setZero(x.size(), Vout.rows());
        //         for (int i = 1; i < node_data_name.size(); i++)
        //             shape_vel.row(i - 1) = Eigen::Map<RowVectorNd>(node_data[i].data(), Vout.rows());
        //     }
        // }
        // else
        {
            Eigen::VectorXd y;
            if (use_scaling_)
                y = x.head(x.size() - dim_);
            else
                y = x;
            
            std::vector<double> y_vec(y.data(), y.data() + y.size());
            {
                POLYFEM_SCOPED_TIMER("mesh inflation");
                logger().info("isosurface inflator input: {}", y.transpose());
                Eigen::MatrixXd vertex_normals, shape_vel;
                utils::inflate(wire_path_, opts_, y_vec, Vout, Fout, vertex_normals, shape_vel);

                Vout.array().rowwise() *= x.tail(dim_).transpose().array();

                Eigen::VectorXd norms = vertex_normals.rowwise().norm();
                boundary_flags.setZero(norms.size());
                for (int i = 0; i < norms.size(); i++)
                    if (norms(i) > 0.1)
                        boundary_flags(i) = true;
                
                shape_velocity.setZero(shape_vel.rows(), vertex_normals.size());
                for (int d = 0; d < dim_; d++)
                    for (int i = 0; i < vertex_normals.rows(); i++)
                        for (int q = 0; q < shape_vel.rows(); q++)
                            shape_velocity(q, dim_ * i + d) = shape_vel(q, i) * vertex_normals(i, d);
            }
            
            write_msh(out_path_, Vout, Fout);
            if (volume_velocity_)
                extend_to_internal();
        }

        last_x = x;

        return true;
    }
    void SDF2Mesh::extend_to_internal() const
    {
        json args = R"(
        {
            "geometry": [
                {
                    "mesh": "",
                    "surface_selection": {
                        "threshold": 1e-7
                    }
                }
            ],
            "space": {
                "discr_order": 1
            },
            "solver": {
                "linear": {
                    "solver": "Eigen::PardisoLDLT"
                }
            },
            "boundary_conditions": {
                "dirichlet_boundary": [
                    {
                        "id": 7,
                        "value": 0.0
                    }
                ],
                "periodic_boundary": [true, true]
            },
            "output": {
                "log": {
                    "level": "info"
                }
            },
            "materials": {
                "type": "Laplacian"
            }
        })"_json;
        
        args["geometry"][0]["mesh"] = out_path_;

        const int dim = Vout.cols();

        Eigen::MatrixXd extended_velocity;
        extended_velocity.setZero(shape_velocity.rows(), shape_velocity.cols());

        State state;
        state.init(args, false);

        logger().info("Laplacian solve for volume shape velocity...");

        state.load_mesh();

        if (state.mesh == nullptr)
            log_and_throw_error("Invalid mesh!");
        
        state.stats.compute_mesh_stats(*state.mesh);

        state.build_basis();

        state.assemble_rhs();
        state.assemble_stiffness_mat();

        state.boundary_nodes.clear();
        std::vector<int> primitive_to_node = state.primitive_to_node();
        std::vector<int> node_to_primitive = state.node_to_primitive();
        for (int i = 0; i < boundary_flags.size(); i++)
            if (boundary_flags(i))
                state.boundary_nodes.push_back(primitive_to_node[i]);

        auto solver = polysolve::LinearSolver::create(state.args["solver"]["linear"]["solver"], state.args["solver"]["linear"]["precond"]);
        solver->setParameters(state.args["solver"]["linear"]);

        StiffnessMatrix A = state.stiffness;
        std::vector<int> boundary_nodes_tmp = state.boundary_nodes;
        {
            const int full_size = A.rows();
            int precond_num = full_size;

            state.full_to_periodic(boundary_nodes_tmp);
            precond_num = state.full_to_periodic(A);

            StiffnessMatrix Atmp = A;
            prefactorize(*solver, Atmp, boundary_nodes_tmp, precond_num, state.args["output"]["data"]["stiffness_mat"]);
        }

        Eigen::MatrixXd rhs;
        rhs.setZero(state.ndof(), shape_velocity.rows() * dim);
        for (int q = 0; q < shape_velocity.rows(); q++)
            for (int d = 0; d < dim; d++)
                for (int i : state.boundary_nodes)
                    rhs(i, q * dim + d) = shape_velocity(q, node_to_primitive[i] * dim + d);

        state.full_to_periodic(rhs, true);
    
        // enforce dirichlet boundary on rhs
        {
            Eigen::VectorXd N;
            N.setZero(A.rows());
            N(boundary_nodes_tmp).setOnes();
            rhs -= ((1.0 - N.array()).matrix()).asDiagonal() * (A * (N.asDiagonal() * rhs));
        }

        for (int q = 0; q < shape_velocity.rows(); q++)
        {
            Eigen::MatrixXd sol(state.ndof(), dim);
            for (int d = 0; d < dim; d++)
            {
                Eigen::VectorXd x;
                x.setZero(rhs.rows());
                solver->solve(rhs.col(q * dim + d), x);

                sol.col(d) = state.periodic_to_full(state.ndof(), x);
            }
            extended_velocity.row(q) = utils::flatten(sol(primitive_to_node, Eigen::all)).transpose();

            // io::VTUWriter writer;
            // writer.add_field("volume_velocity", utils::unflatten(extended_velocity.row(q).transpose(), dim));
            // writer.add_field("boundary_velocity", utils::unflatten(shape_velocity.row(q).transpose(), dim));

            // writer.write_mesh("debug_" + std::to_string(q) + ".vtu", Vout, Fout);
        }
        
        double error = 0;
        for (int i = 0; i < extended_velocity.cols(); i++)
            if (boundary_flags(i / dim))
                error += (extended_velocity.col(i) - shape_velocity.col(i)).norm();
        logger().info("Error of volume shape velocity: {}", error);
        
        std::swap(extended_velocity, shape_velocity);
    }
    int SDF2Mesh::size(const int x_size) const
    {
        return Vout.size();
    }
    Eigen::VectorXd SDF2Mesh::eval(const Eigen::VectorXd &x) const
    {
        if (!isosurface_inflator(x))
        {
            logger().error("Failed to inflate mesh!");
            return Eigen::VectorXd();
        }

        return utils::flatten(Vout);
    } 
    Eigen::VectorXd SDF2Mesh::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
    {
        // assert(x.size() == shape_vel.rows());
        // const int dim = vertex_normals.cols();
        
        Eigen::VectorXd mapped_grad(x.size());
        if (use_scaling_)
        {
            Eigen::VectorXd scale = x.tail(dim_);
            const int n_nodes = shape_velocity.cols() / dim_;
            mapped_grad.head(x.size() - dim_) = shape_velocity * scale.replicate(n_nodes, 1).asDiagonal() * grad;

            Eigen::MatrixXd unflattened_grad = utils::unflatten(grad, dim_);
            mapped_grad.tail(dim_) = (Vout.array() * unflattened_grad.array()).colwise().sum().array() / scale.transpose().array();
        }
        else
            mapped_grad = shape_velocity * grad; // shape_vel * (vertex_normals.array() * utils::unflatten(grad, dim).array()).matrix().rowwise().sum();

        // debug
        // {
        //     static int debug_id = 0;
        //     Eigen::VectorXd grad_normal_coeffs = (vertex_normals.array() * utils::unflatten(grad, dim).array()).matrix().rowwise().sum();
        //     Eigen::MatrixXd grad_normal_direction = vertex_normals.array().colwise() * grad_normal_coeffs.array();
        //     Eigen::MatrixXd grad_tangent_direction = utils::unflatten(grad, dim) - grad_normal_direction;
            
        //     io::VTUWriter writer;
        //     writer.add_field("gradient", utils::unflatten(grad, dim));
        //     writer.add_field("gradient_normal", grad_normal_direction);
        //     writer.add_field("gradient_tangent", grad_tangent_direction);

        //     writer.write_mesh("debug_" + std::to_string(debug_id++) + ".vtu", Vout, Fout);
        // }

        return mapped_grad;
    }

    MeshTiling::MeshTiling(const Eigen::VectorXi &nums, const std::string in_path, const std::string out_path): nums_(nums), in_path_(in_path), out_path_(out_path)
    {
        assert(nums_.size() == 2 || nums_.size() == 3);
        assert(nums.minCoeff() > 0);
    }
    int MeshTiling::size(const int x_size) const
    {
        Eigen::MatrixXd vertices;
        Eigen::MatrixXi cells;
        std::vector<std::vector<int>> elements;
        std::vector<std::vector<double>> weights;
        std::vector<int> body_ids;
        io::MshReader::load(out_path_, vertices, cells, elements, weights, body_ids);
        return vertices.size();
    }
    Eigen::VectorXd MeshTiling::eval(const Eigen::VectorXd &x) const
    {
        Eigen::MatrixXd vertices;
        Eigen::MatrixXi cells;
        std::vector<std::vector<int>> elements;
        std::vector<std::vector<double>> weights;
        std::vector<int> body_ids;
        io::MshReader::load(in_path_, vertices, cells, elements, weights, body_ids);
        const int dim = vertices.cols();

        if (x.size() != vertices.size())
            log_and_throw_error("Inconsistent input mesh in tiling!");
        else if ((x - utils::flatten(vertices)).norm() > 1e-6)
        {
            logger().error("Diff in input mesh and x is {}", (x - utils::flatten(vertices)).norm());
            log_and_throw_error("Inconsistent input mesh in tiling!");
        }
        vertices = utils::unflatten(x, vertices.cols());

        Eigen::MatrixXd Vout;
        Eigen::MatrixXi Fout;
        if (!tiling(vertices, cells, Vout, Fout))
        {
            logger().error("Failed to tile mesh!");
            return Eigen::VectorXd();
        }

        return utils::flatten(Vout);
    }
    Eigen::VectorXd MeshTiling::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
    {
        Eigen::MatrixXd vertices;
        Eigen::MatrixXi cells;
        std::vector<std::vector<int>> elements;
        std::vector<std::vector<double>> weights;
        std::vector<int> body_ids;
        io::MshReader::load(out_path_, vertices, cells, elements, weights, body_ids);
        const int dim = vertices.cols();

        if (grad.size() != vertices.size())
            log_and_throw_error("Inconsistent input mesh in tiling jacobian!");

        Eigen::VectorXd reduced_grad;
        reduced_grad.setZero(x.size());
        assert(grad.size() == index_map.size() * dim);
        for (int i = 0; i < index_map.size(); i++)
            reduced_grad(Eigen::seqN(index_map(i)*dim, dim)) += grad(Eigen::seqN(i*dim, dim));

        return reduced_grad;
    }
    bool MeshTiling::tiling(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Eigen::MatrixXd &Vnew, Eigen::MatrixXi &Fnew) const
    {
        if (last_V.size() == V.size() && last_V == V)
        {
            std::vector<std::vector<int>> elements;
            std::vector<std::vector<double>> weights;
            std::vector<int> body_ids;
            io::MshReader::load(out_path_, Vnew, Fnew, elements, weights, body_ids);
            return true;
        }

        assert(nums_.size() == V.cols());

        Eigen::MatrixXd Vtmp;
        Eigen::MatrixXi Ftmp;
        Vtmp.setZero(V.rows() * nums_.prod(), V.cols());
        Ftmp.setZero(F.rows() * nums_.prod(), F.cols());

        Eigen::MatrixXd bbox(V.cols(), 2);
        bbox.col(0) = V.colwise().minCoeff();
        bbox.col(1) = V.colwise().maxCoeff();
        Eigen::VectorXd size = bbox.col(1) - bbox.col(0);

        if (V.cols() == 2)
        {
            for (int i = 0, idx = 0; i < nums_(0); i++)
            {
                for (int j = 0; j < nums_(1); j++)
                {
                    Vtmp.middleRows(idx * V.rows(), V.rows()) = V;
                    Vtmp.block(idx * V.rows(), 0, V.rows(), 1).array() += size(0) * i;
                    Vtmp.block(idx * V.rows(), 1, V.rows(), 1).array() += size(1) * j;

                    Ftmp.middleRows(idx * F.rows(), F.rows()) = F.array() + idx * V.rows();
                    idx += 1;
                }
            }
        }
        else
        {
            log_and_throw_error("Not implemented!");
        }

        // clean duplicated vertices
        const double eps = 1e-6;
        Eigen::VectorXi indices;
        {
            std::vector<int> tmp;
            for (int i = 0; i < V.rows(); i++)
            {
                Eigen::VectorXd p = V.row(i);
                if ((p - bbox.col(0)).array().abs().minCoeff() < eps || (p - bbox.col(1)).array().abs().minCoeff() < eps)
                    tmp.push_back(i);
            }

            indices.resize(tmp.size() * nums_.prod());
            for (int i = 0; i < nums_.prod(); i++)
            {
                indices.segment(i * tmp.size(), tmp.size()) = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(tmp.data(), tmp.size());
                indices.segment(i * tmp.size(), tmp.size()).array() += i * V.rows();
            }
        }

        Eigen::VectorXi potentially_duplicate_mask(Vtmp.rows());
        potentially_duplicate_mask.setZero();
        potentially_duplicate_mask(indices).array() = 1;
        Eigen::MatrixXd candidates = Vtmp(indices, Eigen::all);

        Eigen::VectorXi SVI;
        std::vector<int> SVJ;
        SVI.setConstant(Vtmp.rows(), -1);
        int id = 0;
        for (int i = 0; i < Vtmp.rows(); i++)
        {
            if (SVI[i] < 0)
            {
                SVI[i] = id;
                SVJ.push_back(i);
                if (potentially_duplicate_mask(i))
                {
                    Eigen::VectorXd diffs = (candidates.rowwise() - Vtmp.row(i)).rowwise().norm();
                    for (int j = 0; j < diffs.size(); j++)
                        if (diffs(j) < eps)
                            SVI[indices[j]] = id;
                }
                id++;
            }
        }
        Vnew = Vtmp(SVJ, Eigen::all);

        index_map.setConstant(Vtmp.rows(), -1);
        for (int i = 0; i < V.rows(); i++)
            for (int j = 0; j < nums_.prod(); j++)
                index_map(j * V.rows() + i) = i;
        index_map = index_map(SVJ).eval();

        Fnew.resizeLike(Ftmp);
        for (int d = 0; d < Ftmp.cols(); d++)
            Fnew.col(d) = SVI(Ftmp.col(d));

        {
            write_msh(out_path_, Vnew, Fnew);

            logger().info("Saved tiled mesh to {}", out_path_);
        }

        last_V = V;
        return true;
    }

    MeshAffine::MeshAffine(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, const std::string in_path, const std::string out_path): A_(A), b_(b), in_path_(in_path), out_path_(out_path)
    {
    }

    Eigen::VectorXd MeshAffine::eval(const Eigen::VectorXd &x) const
    {
        auto V = utils::unflatten(x, A_.rows());
        V = ((V * A_.transpose()).eval().rowwise() + b_.transpose()).eval();

        if (last_x.size() == x.size() && last_x == x)
            return utils::flatten(V);

        // save to file
        {
            Eigen::MatrixXd vertices;
            Eigen::MatrixXi cells;
            std::vector<std::vector<int>> elements;
            std::vector<std::vector<double>> weights;
            std::vector<int> body_ids;
            io::MshReader::load(in_path_, vertices, cells, elements, weights, body_ids);

            write_msh(out_path_, V, cells);

            logger().info("Saved affined mesh to {}", out_path_);
        }

        last_x = x;
        
        return utils::flatten(V);
    }
    Eigen::VectorXd MeshAffine::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
    {
        auto tmp = utils::unflatten(grad, A_.rows());
        return utils::flatten(tmp * A_.transpose());
    }

    PeriodicMeshToMesh::PeriodicMeshToMesh(const Eigen::MatrixXd &V)
    {
        dim = V.cols();

        for (auto &list : periodic_dependence)
            list.clear();

        assert(dim == V.cols());
        const int n_verts = V.rows();

        Eigen::VectorXd min = V.colwise().minCoeff();
        Eigen::VectorXd max = V.colwise().maxCoeff();
        Eigen::VectorXd scale_ = max - min;

        n_periodic_dof = 0;
        dependent_map.resize(n_verts);
        dependent_map.setConstant(-1);

        const double eps = 1e-4 * scale_.maxCoeff();
        Eigen::VectorXi boundary_indices;
        {
            Eigen::VectorXi boundary_mask1 = ((V.rowwise() - min.transpose()).rowwise().minCoeff().array() < eps).select(Eigen::VectorXi::Ones(V.rows()), Eigen::VectorXi::Zero(V.rows()));
            Eigen::VectorXi boundary_mask2 = ((V.rowwise() - max.transpose()).rowwise().maxCoeff().array() > -eps).select(Eigen::VectorXi::Ones(V.rows()), Eigen::VectorXi::Zero(V.rows()));
            Eigen::VectorXi boundary_mask = boundary_mask1.array() + boundary_mask2.array();

            boundary_indices.setZero(boundary_mask.sum());
            for (int i = 0, j = 0; i < boundary_mask.size(); i++)
                if (boundary_mask[i])
                    boundary_indices[j++] = i;
        }

        // find corresponding periodic boundary nodes
        Eigen::MatrixXd V_boundary = V(boundary_indices, Eigen::all);
        for (int d = 0; d < dim; d++)
        {
            Eigen::VectorXi mask1 = (V_boundary.col(d).array() < min(d) + eps).select(Eigen::VectorXi::Ones(V_boundary.rows()), Eigen::VectorXi::Zero(V_boundary.rows()));
            Eigen::VectorXi mask2 = (V_boundary.col(d).array() > max(d) - eps).select(Eigen::VectorXi::Ones(V_boundary.rows()), Eigen::VectorXi::Zero(V_boundary.rows()));

            for (int i = 0; i < mask1.size(); i++)
            {
                if (!mask1(i))
                    continue;
                
                bool found_target = false;
                for (int j = 0; j < mask2.size(); j++)
                {
                    if (!mask2(j))
                        continue;
                    
                    RowVectorNd projected_diff = V_boundary.row(j) - V_boundary.row(i);
                    projected_diff(d) = 0;
                    if (projected_diff.norm() < eps)
                    {
                        dependent_map(boundary_indices[j]) = boundary_indices[i];
                        std::array<int, 2> pair = {{boundary_indices[i], boundary_indices[j]}};
                        periodic_dependence[d].push_back(pair);
                        found_target = true;
                        break;
                    }
                }
                if (!found_target)
                    throw std::runtime_error("Periodic mesh failed to find corresponding nodes!");
            }
        }

        // break dependency chains into direct dependency
        for (int d = 0; d < dim; d++)
            for (int i = 0; i < dependent_map.size(); i++)
                if (dependent_map(i) >= 0 && dependent_map(dependent_map(i)) >= 0)
                    dependent_map(i) = dependent_map(dependent_map(i));
        
        Eigen::VectorXi reduce_map;
        reduce_map.setZero(dependent_map.size());
        for (int i = 0; i < dependent_map.size(); i++)
            if (dependent_map(i) < 0)
                reduce_map(i) = n_periodic_dof++;
        for (int i = 0; i < dependent_map.size(); i++)
            if (dependent_map(i) >= 0)
                reduce_map(i) = reduce_map(dependent_map(i));
        
        dependent_map = std::move(reduce_map);
    }

    Eigen::VectorXd PeriodicMeshToMesh::eval(const Eigen::VectorXd &x) const
    {
        assert(x.size() == input_size());

        Eigen::VectorXd scale = x.tail(dim);
        Eigen::VectorXd y;
        y.setZero(size(x.size()));
        for (int i = 0; i < dependent_map.size(); i++)
            y.segment(i * dim, dim) = x.segment(dependent_map(i) * dim, dim).array() * scale.array();
        
        for (int d = 0; d < dim; d++)
        {
            const auto &dependence_list = periodic_dependence[d];
            for (const auto &pair : dependence_list)
                y(pair[1] * dim + d) += scale[d];
        }

        return y;
    }

    Eigen::VectorXd PeriodicMeshToMesh::inverse_eval(const Eigen::VectorXd &y)
    {
        assert(y.size() == dim * dependent_map.size());
        Eigen::VectorXd x;
        x.setZero(input_size());

        Eigen::MatrixXd V = utils::unflatten(y, dim);
        Eigen::VectorXd min = V.colwise().minCoeff();
        Eigen::VectorXd max = V.colwise().maxCoeff();
        Eigen::VectorXd scale = max - min;
        x.tail(dim) = scale;

        Eigen::VectorXd z = y;
        for (int d = 0; d < dim; d++)
        {
            const auto &dependence_list = periodic_dependence[d];
            for (const auto &pair : dependence_list)
                z(pair[1] * dim + d) -= scale[d];
        }

        for (int i = 0; i < dependent_map.size(); i++)
            x.segment(dependent_map(i) * dim, dim) = z.segment(i * dim, dim).array() / scale.array();

        return x;
    }

    Eigen::VectorXd PeriodicMeshToMesh::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
    {
        assert(x.size() == input_size());
        Eigen::VectorXd reduced_grad;
        reduced_grad.setZero(x.size());

        for (int i = 0; i < dependent_map.size(); i++)
            reduced_grad.segment(dependent_map(i) * dim, dim).array() += grad.segment(i * dim, dim).array() * x.tail(dim).array();
        
        for (int i = 0; i < dependent_map.size(); i++)
            reduced_grad.segment(dim * n_periodic_dof, dim).array() += grad.segment(i * dim, dim).array() * x.segment(dependent_map(i) * dim, dim).array();

        for (int d = 0; d < dim; d++)
        {
            const auto &dependence_list = periodic_dependence[d];
            for (const auto &pair : dependence_list)
                reduced_grad(dim * n_periodic_dof + d) += grad(pair[1] * dim + d);
        }

        return reduced_grad;
    }
}
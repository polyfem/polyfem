#include "SDFParametrizations.hpp"

#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/io/MshReader.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <igl/writeMSH.h>

namespace polyfem::solver
{
    namespace {
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

        std::string shape_params = "--params \"";
        for (int i = 0; i < x.size(); i++)
            shape_params += to_string_with_precision(x(i), 16) + " ";
        shape_params += "\" ";

        std::string command = inflator_path_ + " " + shape_params + " -S " + sdf_velocity_path_ + " " + msh_path_;

        int return_val;
        try 
        {
            return_val = system(command.c_str());
        }
        catch (const std::exception &err)
        {
            logger().error("remesh command \"{}\" returns {}", command, return_val);

            return false;
        }

        logger().info("remesh command \"{}\" returns {}", command, return_val);

        last_x = x;

        return true;
    }
    int SDF2Mesh::size(const int x_size) const
    {
        Eigen::MatrixXd vertices;
        Eigen::MatrixXi cells;
        std::vector<std::vector<int>> elements;
        std::vector<std::vector<double>> weights;
        std::vector<int> body_ids;
        io::MshReader::load(sdf_velocity_path_, vertices, cells, elements, weights, body_ids);
        return vertices.size();
    }
    Eigen::VectorXd SDF2Mesh::eval(const Eigen::VectorXd &x) const
    {
        if (!isosurface_inflator(x))
        {
            logger().error("Failed to inflate mesh!");
            return Eigen::VectorXd();
        }

        Eigen::MatrixXd vertices;
        Eigen::MatrixXi cells;
        std::vector<std::vector<int>> elements;
        std::vector<std::vector<double>> weights;
        std::vector<int> body_ids;
        io::MshReader::load(sdf_velocity_path_, vertices, cells, elements, weights, body_ids);
        const int dim = vertices.cols();

        return utils::flatten(vertices);
    } 
    Eigen::VectorXd SDF2Mesh::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
    {
        if (!isosurface_inflator(x))
        {
            logger().error("Failed to inflate mesh!");
            return Eigen::VectorXd();
        }

        Eigen::MatrixXd vertices;
        Eigen::MatrixXi cells;
        std::vector<std::vector<int>> elements;
        std::vector<std::vector<double>> weights;
        std::vector<int> body_ids;
        std::vector<std::string> node_data_name;
        std::vector<std::vector<double>> node_data;
        io::MshReader::load(sdf_velocity_path_, vertices, cells, elements, weights, body_ids, node_data_name, node_data);
        const int dim = vertices.cols();
        
        assert(node_data_name.size() == node_data.size());

        Eigen::VectorXd mapped_grad;
        mapped_grad.setZero(node_data.size() - 1);
        for (int j = 0; j < vertices.rows(); j++)
            for (int i = 1; i < node_data_name.size(); i++)
                for (int d = 0; d < dim; d++)
                    mapped_grad(i - 1) += node_data[0][j * 3 + d] * node_data[i][j] * grad(j * dim + d);

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
        if (last_x.size() == V.size() && last_x == V)
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
            Eigen::MatrixXd Vsave;
            Vsave.setZero(Vnew.rows(), 3);
            Vsave.leftCols(Vnew.cols()) = Vnew;

            const int dim = Vnew.cols();
            Eigen::MatrixXi Tri = (dim == 3) ? Eigen::MatrixXi() : Fnew;
            Eigen::MatrixXi Tet = (dim == 3) ? Fnew : Eigen::MatrixXi();

            igl::writeMSH(out_path_, Vsave, Tri, Tet, Eigen::MatrixXi::Zero(Tri.rows(), 1), Eigen::MatrixXi::Zero(Tet.rows(), 1), std::vector<std::string>(), std::vector<Eigen::MatrixXd>(), std::vector<std::string>(), std::vector<Eigen::MatrixXd>(), std::vector<Eigen::MatrixXd>());

            logger().info("Saved tiled mesh to {}", out_path_);
        }

        last_x = V;
        return true;
    }
}
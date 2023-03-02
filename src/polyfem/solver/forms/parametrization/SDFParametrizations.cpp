#include "SDFParametrizations.hpp"

#include <polyfem/utils/IsosurfaceInflator.hpp>
#include <polyfem/io/MshReader.hpp>
#include <polyfem/utils/Timer.hpp>
#include <igl/writeMSH.h>

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
    }

    bool SDF2Mesh::isosurface_inflator(const Eigen::VectorXd &x) const
    {
        if (last_x.size() == x.size() && last_x == x)
            return true;

        double inflation_time = 0, saving_time = 0;

        std::vector<double> x_vec(x.data(), x.data() + x.size());
        {
            POLYFEM_SCOPED_TIMER("mesh inflation", inflation_time);
            utils::inflate(wire_path_, opts_, x_vec, Vout, Fout, vertex_normals, shape_vel);
        }
        
        write_msh(out_path_, Vout, Fout);

        last_x = x;

        return true;
    }
    int SDF2Mesh::size(const int x_size) const
    {
        if (last_x.size() == x_size)
            return Vout.size();
        else
            return 0;
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
        if (!isosurface_inflator(x))
        {
            logger().error("Failed to inflate mesh!");
            return Eigen::VectorXd();
        }
        
        assert(x.size() == shape_vel.rows());

        const int dim = vertex_normals.cols();
        
        Eigen::VectorXd mapped_grad  = shape_vel * (vertex_normals.array() * utils::unflatten(grad, dim).array()).matrix().rowwise().sum();

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
        
        return utils::flatten(V);
    }
    Eigen::VectorXd MeshAffine::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
    {
        auto tmp = utils::unflatten(grad, A_.rows());
        return utils::flatten(tmp * A_.transpose());
    }
}
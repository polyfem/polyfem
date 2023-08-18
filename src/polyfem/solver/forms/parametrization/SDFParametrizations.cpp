#include "SDFParametrizations.hpp"

#include <polyfem/io/MshReader.hpp>
#include <polyfem/utils/Timer.hpp>
#include <igl/writeMSH.h>
#include <paraviewo/VTUWriter.hpp>

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

		double triangle_jacobian(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3)
		{
			Eigen::VectorXd a = v2 - v1, b = v3 - v1;
			return a(0) * b(1) - b(0) * a(1);
		}

		double tet_determinant(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3, const Eigen::VectorXd &v4)
		{
			Eigen::Matrix3d mat;
			mat.col(0) << v2 - v1;
			mat.col(1) << v3 - v1;
			mat.col(2) << v4 - v1;
			return mat.determinant();
		}

		bool is_flipped(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
		{
			if (F.cols() == 3)
			{
				for (int i = 0; i < F.rows(); i++)
					if (triangle_jacobian(V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2))) <= 0)
						return true;
			}
			else if (F.cols() == 4)
			{
				for (int i = 0; i < F.rows(); i++)
					if (tet_determinant(V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2)), V.row(F(i, 3))) <= 0)
						return true;
			}
			else
			{
				return true;
			}

			return false;
		}
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
        const double eps = 1e-4;
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
        dim_ = V.cols();

        for (auto &list : periodic_dependence)
            list.clear();

        assert(dim_ == V.cols());
        const int n_verts = V.rows();

        Eigen::VectorXd min = V.colwise().minCoeff();
        Eigen::VectorXd max = V.colwise().maxCoeff();
        Eigen::VectorXd scale_ = max - min;

        n_periodic_dof_ = 0;
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
        for (int d = 0; d < dim_; d++)
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
                        periodic_dependence[d].insert(pair);
                        found_target = true;
                        break;
                    }
                }
                if (!found_target)
                    throw std::runtime_error("Periodic mesh failed to find corresponding nodes!");
            }
        }

        // break dependency chains into direct dependency
        for (int d = 0; d < dim_; d++)
            for (int i = 0; i < dependent_map.size(); i++)
                if (dependent_map(i) >= 0 && dependent_map(dependent_map(i)) >= 0)
                    dependent_map(i) = dependent_map(dependent_map(i));
        
        Eigen::VectorXi reduce_map;
        reduce_map.setZero(dependent_map.size());
        for (int i = 0; i < dependent_map.size(); i++)
            if (dependent_map(i) < 0)
                reduce_map(i) = n_periodic_dof_++;
        for (int i = 0; i < dependent_map.size(); i++)
            if (dependent_map(i) >= 0)
                reduce_map(i) = reduce_map(dependent_map(i));
        
        dependent_map = std::move(reduce_map);
    }

    Eigen::VectorXd PeriodicMeshToMesh::eval(const Eigen::VectorXd &x) const
    {
        assert(x.size() == input_size());

        Eigen::VectorXd scale = x.tail(dim_);
        Eigen::VectorXd y;
        y.setZero(size(x.size()));
        for (int i = 0; i < dependent_map.size(); i++)
            y.segment(i * dim_, dim_) = x.segment(dependent_map(i) * dim_, dim_).array() * scale.array();
        
        for (int d = 0; d < dim_; d++)
        {
            const auto &dependence_list = periodic_dependence[d];
            for (const auto &pair : dependence_list)
                y(pair[1] * dim_ + d) += scale[d];
        }

        return y;
    }

    Eigen::VectorXd PeriodicMeshToMesh::inverse_eval(const Eigen::VectorXd &y)
    {
        assert(y.size() == dim_ * dependent_map.size());
        Eigen::VectorXd x;
        x.setZero(input_size());

        Eigen::MatrixXd V = utils::unflatten(y, dim_);
        Eigen::VectorXd min = V.colwise().minCoeff();
        Eigen::VectorXd max = V.colwise().maxCoeff();
        Eigen::VectorXd scale = max - min;
        x.tail(dim_) = scale;

        Eigen::VectorXd z = y;
        for (int d = 0; d < dim_; d++)
        {
            const auto &dependence_list = periodic_dependence[d];
            for (const auto &pair : dependence_list)
                z(pair[1] * dim_ + d) -= scale[d];
        }

        for (int i = 0; i < dependent_map.size(); i++)
            x.segment(dependent_map(i) * dim_, dim_) = z.segment(i * dim_, dim_).array() / scale.array();

        if ((y - eval(x)).norm() > 1e-5)
        {
            std::cout << (y - eval(x)).transpose() << "\n";
            log_and_throw_error("Bug in periodic mesh!");
        }

        return x;
    }

    Eigen::VectorXd PeriodicMeshToMesh::apply_jacobian(const Eigen::VectorXd &grad, const Eigen::VectorXd &x) const
    {
        assert(x.size() == input_size());
        Eigen::VectorXd reduced_grad;
        reduced_grad.setZero(x.size());

        for (int i = 0; i < dependent_map.size(); i++)
            reduced_grad.segment(dependent_map(i) * dim_, dim_).array() += grad.segment(i * dim_, dim_).array() * x.tail(dim_).array();
        
        for (int i = 0; i < dependent_map.size(); i++)
            reduced_grad.segment(dim_ * n_periodic_dof_, dim_).array() += grad.segment(i * dim_, dim_).array() * x.segment(dependent_map(i) * dim_, dim_).array();

        for (int d = 0; d < dim_; d++)
        {
            const auto &dependence_list = periodic_dependence[d];
            for (const auto &pair : dependence_list)
                reduced_grad(dim_ * n_periodic_dof_ + d) += grad(pair[1] * dim_ + d);
        }

        return reduced_grad;
    }
}
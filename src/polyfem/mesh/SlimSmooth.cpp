#include "SlimSmooth.hpp"

#include <igl/boundary_facets.h>
#include <polyfem/utils/Logger.hpp>
#include <igl/slim.h>

namespace polyfem::mesh
{
	namespace
	{
		double triangle_jacobian(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2, const Eigen::VectorXd &v3)
		{
			Eigen::VectorXd a = v2 - v1, b = v3 - v1;
			return a(0) * b(1) - b(0) * a(1);
		}

		double tet_determinant(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const Eigen::Vector3d &v3, const Eigen::Vector3d &v4)
		{
			Eigen::Matrix3d mat;
            mat <<
                v2 - v1,
                v3 - v1,
                v4 - v1;
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
			    log_and_throw_adjoint_error("Invalid element type for Jacobian determinant!");
			
            return false;
		}

	} // namespace

	bool apply_slim(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &V_new, Eigen::MatrixXd &V_smooth, const int max_iters)
	{
		const int dim = F.cols() - 1;

        if (is_flipped(V_new, F))
        {
            adjoint_logger().warn("Mesh is flipped before SLIM!");
            return false;
        }

        Eigen::MatrixXd V_extended;
        V_extended.setZero(V.rows(), 3);
        V_extended.leftCols(dim) = V;

        Eigen::VectorXi boundary_indices;
        {
            Eigen::MatrixXi boundary;
            igl::boundary_facets(F, boundary);
            std::set<int> slim_constrained_nodes;
            for (int i = 0; i < boundary.rows(); ++i)
                for (int j = 0; j < boundary.cols(); ++j)
                    slim_constrained_nodes.insert(boundary(i, j));

            
            boundary_indices.setZero(slim_constrained_nodes.size());
            int i = 0;
            for (const auto &c : slim_constrained_nodes)
                boundary_indices(i++) = c;
        }

        const double soft_const_p = 1e5;
        const int slim_iters = 2;
        const double tol = 1e-8;

        const Eigen::MatrixXd boundary_constraints = V_new(boundary_indices, Eigen::all);

        igl::SLIMData slim_data;
        slim_data.exp_factor = 5;
        igl::slim_precompute(
            V_extended,
            F,
            V,
            slim_data,
            igl::SYMMETRIC_DIRICHLET,
            boundary_indices,
            boundary_constraints,
            soft_const_p);

        V_smooth.setZero(V.rows(), V.cols());

        auto is_good_enough = [](const Eigen::MatrixXd &V, const Eigen::VectorXi &b, const Eigen::MatrixXd &C, double &error, double tol = 1e-6) {
            error = 0.0;

            for (unsigned i = 0; i < b.rows(); i++)
                error += (C.row(i) - V.row(b(i))).squaredNorm();

            return error < tol;
        };

        double error = 0;
        int it = 0;
        bool good_enough = false;

        do
        {
            igl::slim_solve(slim_data, slim_iters);
            // good_enough = is_good_enough(slim_data.V_o, boundary_indices, boundary_constraints, error, 1e-8);
            error = (slim_data.V_o(boundary_indices, Eigen::all) - boundary_constraints).squaredNorm();
            good_enough = error < 1e-8;
            V_smooth = slim_data.V_o.leftCols(dim);
            it += slim_iters;
        } while (it < max_iters && !good_enough);

        V_smooth(boundary_indices, Eigen::all) = boundary_constraints;
        logger().debug("SLIM finished in {} iterations", it);

        if (good_enough)
            logger().debug("SLIM succeeded.");
        else
            logger().warn("SLIM cannot smooth correctly. Error: {}", error);
        
        return good_enough;
	}
} // namespace polyfem::mesh

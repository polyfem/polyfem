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

	} // namespace

	void apply_slim(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, Eigen::MatrixXd &V_new)
	{
		const int dim = F.cols() - 1;
		if (dim == 2)
		{
			logger().error("SLIM smoothing not implemented for 2d");
			assert(false);
		}
		else
		{
			Eigen::MatrixXi boundary;
			igl::boundary_facets(F, boundary);
			std::set<int> slim_constrained_nodes;
			for (int i = 0; i < boundary.rows(); ++i)
				for (int j = 0; j < boundary.cols(); ++j)
					slim_constrained_nodes.insert(boundary(i, j));
			std::vector<int> boundary_indices;
			for (const auto &c : slim_constrained_nodes)
				boundary_indices.push_back(c);

			const int dim = F.cols() - 1;
			const double soft_const_p = 1e5;
			const int min_iter = 2;
			int max_it = dim == 2 ? 20 : 50;
			const double tol = 1e-8;

			igl::SLIMData slim_data;
			slim_data.exp_factor = 5;
			Eigen::MatrixXd V_extended;
			V_extended.setZero(V.rows(), 3);
			V_extended.block(0, 0, V.rows(), dim) = V;
			Eigen::VectorXi boundary_indices_ = Eigen::VectorXi::Map(boundary_indices.data(), boundary_indices.size());
			Eigen::MatrixXd boundary_constraints = V_extended(boundary_indices_, Eigen::all);
			igl::slim_precompute(
				V_extended,
				F,
				V,
				slim_data,
				igl::SYMMETRIC_DIRICHLET,
				boundary_indices_,
				boundary_constraints,
				soft_const_p);

			V_new.setZero(V.rows(), V.cols());

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
				igl::slim_solve(slim_data, min_iter);
				good_enough = is_good_enough(slim_data.V_o, boundary_indices_, boundary_constraints, error, 1e-8);
				V_new = slim_data.V_o.block(0, 0, V_new.rows(), dim);
				it += min_iter;
			} while (it < max_it && !good_enough);

			for (unsigned i = 0; i < boundary_indices_.rows(); i++)
				V_new.row(boundary_indices_(i)) = boundary_constraints.row(i);

			logger().debug("SLIM finished in {} iterations", it);

			if (!good_enough)
				logger()
					.warn("Slimflator could not inflate correctly. Error: {}", error);

			logger().debug("SLIM succeeded.");
		}
	}
} // namespace polyfem::mesh
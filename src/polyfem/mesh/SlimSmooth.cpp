#include "SlimSmooth.hpp"

#include <igl/boundary_facets.h>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/solver/AdjointTools.hpp>
#include <igl/slim.h>

namespace polyfem::mesh
{

	bool apply_slim(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &V_new, Eigen::MatrixXd &V_smooth, const int max_iters)
	{
		const int dim = F.cols() - 1;
		if (dim == 2)
		{
			logger().error("SLIM smoothing not implemented for 2d");
			return true;
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
			const int slim_iters = 2;
			const double tol = 1e-8;

			igl::SLIMData slim_data;
			slim_data.exp_factor = 5;
			Eigen::MatrixXd V_extended;
			V_extended.setZero(V.rows(), 3);
			V_extended.block(0, 0, V.rows(), dim) = V;
			Eigen::VectorXi boundary_indices_ = Eigen::VectorXi::Map(boundary_indices.data(), boundary_indices.size());
			Eigen::MatrixXd boundary_constraints = V_new(boundary_indices_, Eigen::all);

			igl::slim_precompute(
				V_extended,
				F,
				V,
				slim_data,
				igl::SYMMETRIC_DIRICHLET,
				boundary_indices_,
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
				good_enough = is_good_enough(slim_data.V_o, boundary_indices_, boundary_constraints, error, 1e-8);
				V_smooth = slim_data.V_o.block(0, 0, V_smooth.rows(), dim);
				it += slim_iters;
			} while (it < max_iters && !good_enough);

			for (unsigned i = 0; i < boundary_indices_.rows(); i++)
				V_smooth.row(boundary_indices_(i)) = boundary_constraints.row(i);

			logger().debug("SLIM finished in {} iterations", it);

			if (!good_enough)
			{
				logger()
					.warn("Slimflator could not inflate correctly. Error: {}", error);
				return false;
			}
			else
			{
				logger().debug("SLIM succeeded.");
				return true;
			}
		}
	}
} // namespace polyfem::mesh
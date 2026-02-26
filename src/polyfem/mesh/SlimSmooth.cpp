#include "SlimSmooth.hpp"

#include <ipc/utils/eigen_ext.hpp>
#include <igl/boundary_facets.h>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/GeometryUtils.hpp>
#include <igl/slim.h>

namespace polyfem::mesh
{
	bool apply_slim(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &V_new, Eigen::MatrixXd &V_smooth, const int max_iters)
	{
		const int dim = F.cols() - 1;

		if (utils::is_flipped(V_new, F))
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

		double error = 0;
		int it = 0;
		bool good_enough = false;

		do
		{
			igl::slim_solve(slim_data, slim_iters);
			error = (slim_data.V_o(boundary_indices, Eigen::all) - boundary_constraints).squaredNorm() / boundary_indices.size();
			good_enough = error < 1e-7;
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

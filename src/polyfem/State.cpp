#include <polyfem/State.hpp>

#include <polyfem/utils/Logger.hpp>
#include <polyfem/varforms/VarForm.hpp>

#include <cassert>

namespace polyfem
{
	namespace
	{
		bool iso_parametric(const mesh::Mesh &mesh, const json &args)
		{
			if (mesh.has_poly())
				return true;

			if (args["space"]["basis_type"] == "Bernstein")
				return false;

			if (args["space"]["basis_type"] == "Spline")
				return true;

			if (mesh.is_rational())
				return false;

			if (args["space"]["use_p_ref"])
				return false;

			if (args["boundary_conditions"]["periodic_boundary"]["enabled"].get<bool>())
				return false;

			if (mesh.orders().size() <= 0)
			{
				if (args["space"]["discr_order"] == 1)
					return true;
				else
					return args["space"]["advanced"]["isoparametric"];
			}

			if (mesh.orders().minCoeff() != mesh.orders().maxCoeff())
				return false;

			if (args["space"]["discr_order"] == mesh.orders().minCoeff())
				return true;

			return args["space"]["advanced"]["isoparametric"];
		}
	} // namespace

	void State::build_basis()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}

		mesh->prepare_mesh();
		assert(variational_formulation != nullptr);
		variational_formulation->build_basis(*mesh, iso_parametric(*mesh, args), args);
	}

	void State::assemble_mass_mat()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		assert(variational_formulation != nullptr);
		variational_formulation->assemble_mass_mat(*mesh, args);
	}

	void State::assemble_rhs()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		assert(variational_formulation != nullptr);
		variational_formulation->assemble_rhs(*mesh, args);
	}

	void State::solve_problem(Eigen::MatrixXd &sol)
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}
		assert(variational_formulation != nullptr);

		variational_formulation->solve(sol);
	}

	void State::solve(Eigen::MatrixXd &sol)
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}

		build_basis();
		assemble_rhs();
		assemble_mass_mat();
		solve_problem(sol);
	}
} // namespace polyfem

#include <polyfem/varforms/ShouldUseIsoparametric.hpp>

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/utils/Logger.hpp>

namespace polyfem::varform
{
	bool should_use_isoparametric(const mesh::Mesh &mesh, const json &args)
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
			return args["space"]["advanced"]["isoparametric"];
		}

		if (mesh.orders().minCoeff() != mesh.orders().maxCoeff())
			return false;

		if (args["space"]["discr_order"] == mesh.orders().minCoeff())
			return true;

		return args["space"]["advanced"]["isoparametric"];
	}
} // namespace polyfem::varform

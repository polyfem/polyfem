#include "BuildBasis.hpp"

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/basis/InterfaceData.hpp>
#include <polyfem/basis/LagrangeBasis2d.hpp>
#include <polyfem/basis/LagrangeBasis3d.hpp>
#include <polyfem/basis/PolygonalBasis2d.hpp>
#include <polyfem/basis/PolygonalBasis3d.hpp>
#include <polyfem/basis/SplineBasis2d.hpp>
#include <polyfem/basis/SplineBasis3d.hpp>
#include <polyfem/basis/barycentric/MVPolygonalBasis2d.hpp>
#include <polyfem/basis/barycentric/WSPolygonalBasis2d.hpp>
#include <polyfem/mesh/mesh2D/Mesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/utils/Logger.hpp>

#include <igl/Timer.h>

namespace polyfem::varform
{
	BasisBuildResult build_basis(
		const mesh::Mesh &mesh,
		const json &args,
		const std::string &formulation,
		const Eigen::VectorXi &orders,
		const Eigen::VectorXi &orders_q,
		const int value_dim,
		const bool is_geometry_basis,
		const assembler::LinearAssembler *linear_assembler)
	{
		BasisBuildResult result;
		std::map<int, basis::InterfaceData> polygon_interfaces;

		const int quadrature_order = args["space"]["advanced"]["quadrature_order"].get<int>();
		const int mass_quadrature_order = args["space"]["advanced"]["mass_quadrature_order"].get<int>();
		const bool has_polys = mesh.has_poly();
		const bool use_corner_quadrature = args["space"]["advanced"]["use_corner_quadrature"];

		// ----------------------------------------------
		// Phase 1: Build non-poly basis.
		// ----------------------------------------------

		igl::Timer timer;
		timer.start();
		if (mesh.is_volume())
		{
			const auto &mesh_3d = dynamic_cast<const mesh::Mesh3D &>(mesh);
			if (args["space"]["basis_type"] == "Spline")
			{
				result.n_bases = basis::SplineBasis3d::build_bases(
					mesh_3d, formulation, quadrature_order, mass_quadrature_order,
					result.bases, result.local_boundary, polygon_interfaces);
			}
			else
			{
				result.n_bases = basis::LagrangeBasis3d::build_bases(
					mesh_3d, formulation, quadrature_order, mass_quadrature_order,
					orders, orders_q,
					!is_geometry_basis && args["space"]["basis_type"] == "Bernstein",
					!is_geometry_basis && args["space"]["basis_type"] == "Serendipity",
					has_polys, false, use_corner_quadrature,
					result.bases, result.local_boundary, polygon_interfaces, result.mesh_nodes);
			}
		}
		else
		{
			const auto &mesh_2d = dynamic_cast<const mesh::Mesh2D &>(mesh);
			if (args["space"]["basis_type"] == "Spline")
			{
				result.n_bases = basis::SplineBasis2d::build_bases(
					mesh_2d, formulation, quadrature_order, mass_quadrature_order,
					result.bases, result.local_boundary, polygon_interfaces);
			}
			else
			{
				result.n_bases = basis::LagrangeBasis2d::build_bases(
					mesh_2d, formulation, quadrature_order, mass_quadrature_order,
					orders,
					!is_geometry_basis && args["space"]["basis_type"] == "Bernstein",
					!is_geometry_basis && args["space"]["basis_type"] == "Serendipity",
					has_polys, false, use_corner_quadrature,
					result.bases, result.local_boundary, polygon_interfaces, result.mesh_nodes);
			}
		}
		timer.stop();
		result.standard_basis_time = timer.getElapsedTime();

		if (polygon_interfaces.empty())
			return result;

		// ----------------------------------------------
		// Phase 2: Build poly basis.
		// ----------------------------------------------

		timer.start();
		const std::string polygon_basis_type = args["space"]["poly_basis_type"];
		int new_bases = 0;
		if (mesh.is_volume())
		{
			if (polygon_basis_type == "MeanValue" || polygon_basis_type == "Wachspress")
				logger().error("Barycentric bases not supported in 3D");
			if (linear_assembler == nullptr)
				log_and_throw_error("MFSHarmonic basis requires a linear assembler");

			new_bases = basis::PolygonalBasis3d::build_bases(
				*linear_assembler,
				args["space"]["advanced"]["n_harmonic_samples"],
				dynamic_cast<const mesh::Mesh3D &>(mesh),
				result.n_bases,
				quadrature_order,
				mass_quadrature_order,
				args["space"]["advanced"]["integral_constraints"],
				result.bases, result.bases, polygon_interfaces, result.polyhedra);
		}
		else
		{
			const auto &mesh_2d = dynamic_cast<const mesh::Mesh2D &>(mesh);
			if (polygon_basis_type == "MeanValue")
			{
				new_bases = basis::MVPolygonalBasis2d::build_bases(
					formulation, value_dim, mesh_2d, result.n_bases,
					quadrature_order, mass_quadrature_order,
					result.bases, result.local_boundary, result.polygons);
			}
			else if (polygon_basis_type == "Wachspress")
			{
				new_bases = basis::WSPolygonalBasis2d::build_bases(
					formulation, value_dim, mesh_2d, result.n_bases,
					quadrature_order, mass_quadrature_order,
					result.bases, result.local_boundary, result.polygons);
			}
			else
			{
				if (linear_assembler == nullptr)
					log_and_throw_error("MFSHarmonic basis requires a linear assembler");
				// RBF basis is always iso parametric. This is an invariant across the codebase.
				new_bases = basis::PolygonalBasis2d::build_bases(
					*linear_assembler,
					args["space"]["advanced"]["n_harmonic_samples"],
					mesh_2d, result.n_bases,
					quadrature_order, mass_quadrature_order,
					args["space"]["advanced"]["integral_constraints"],
					result.bases, result.bases, polygon_interfaces, result.polygons);
			}
		}

		timer.stop();
		result.polygon_basis_time = timer.getElapsedTime();
		result.n_bases += new_bases;
		return result;
	}
} // namespace polyfem::varform

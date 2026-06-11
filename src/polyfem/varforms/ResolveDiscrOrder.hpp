#pragma once

#include <polyfem/Common.hpp>

#include <Eigen/Dense>

#include <string>

namespace polyfem
{
	namespace io
	{
		class OutStatsData;
	}

	namespace mesh
	{
		class Mesh;
	}
}

namespace polyfem::varform
{
	struct DiscOrders
	{
		Eigen::VectorXi orders;   // final per-element orders (possibly p-refined)
		Eigen::VectorXi ordersq; // pre-refinement snapshot (for quadrature)
	};

	/// Resolve per-element discretization orders, optionally applying a priori p-refinement.
	///
	/// The order_spec can be:
	///   - An integer: uniform order for all elements.
	///   - A string: path to a file containing per-element orders.
	///   - An array: maps body IDs to orders (missing IDs default to 1).
	///
	/// If args["space"]["use_p_ref"] is true, applies a priori p-refinement
	/// to the resolved orders and snapshots the pre-refinement values in ordersq.
	///
	/// @param args        Full JSON args (used for order spec and p-refinement params)
	/// @param root_path   Root path for resolving relative file paths
	/// @param mesh        The mesh
	/// @param stats       Statistics object (for p-refinement to record angles etc.)
	/// @return            DiscOrders with final orders and pre-refinement snapshot
	DiscOrders resolve_discr_orders(const json &args,
									const std::string &root_path,
									const mesh::Mesh &mesh,
									io::OutStatsData &stats);

	/// Resolve geometry mapping discretization orders.
	///
	/// Always returns a populated vector:
	///   - Isoparametric: returns a copy of disc_orders (geometry shares solution orders)
	///   - Non-isoparametric: returns mesh.orders() if available, otherwise all-ones
	///
	/// @param mesh            The mesh
	/// @param disc_orders     The solution space discretization orders (after p-refinement)
	/// @param isoparametric   Whether the geometry mapping is isoparametric
	/// @return                Per-element geometry discretization order vector
	Eigen::VectorXi resolve_geom_orders(const mesh::Mesh &mesh,
										const Eigen::VectorXi &disc_orders,
										bool isoparametric);
} // namespace polyfem::varform
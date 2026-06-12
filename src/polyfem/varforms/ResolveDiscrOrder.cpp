#include <polyfem/varforms/ResolveDiscrOrder.hpp>

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/OutStatsData.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/refinement/APriori.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/StringUtils.hpp>

namespace polyfem::varform
{
	DiscOrders resolve_discr_orders(const json &args,
									const std::string &root_path,
									const mesh::Mesh &mesh,
									io::OutStatsData &stats)
	{
		const json &order_spec = args["space"]["discr_order"];

		Eigen::VectorXi disc_orders(mesh.n_elements());

		if (order_spec.is_number_integer())
		{
			disc_orders.setConstant(order_spec);
		}
		else if (order_spec.is_string())
		{
			const std::string discr_orders_path = utils::resolve_path(order_spec, root_path);
			Eigen::MatrixXi tmp;
			polyfem::io::read_matrix(discr_orders_path, tmp);
			assert(tmp.size() == disc_orders.size());
			assert(tmp.cols() == 1);
			disc_orders = tmp;
		}
		else if (order_spec.is_array())
		{
			const auto b_discr_orders = order_spec;

			std::map<int, int> b_orders;
			for (size_t i = 0; i < b_discr_orders.size(); ++i)
			{
				assert(b_discr_orders[i]["id"].is_array() || b_discr_orders[i]["id"].is_number_integer());

				const int order = b_discr_orders[i]["order"];
				for (const int id : polyfem::utils::json_as_array<int>(b_discr_orders[i]["id"]))
				{
					b_orders[id] = order;
					logger().trace("bid {}, discr {}", id, order);
				}
			}

			for (int e = 0; e < mesh.n_elements(); ++e)
			{
				const int bid = mesh.get_body_id(e);
				const auto order = b_orders.find(bid);
				if (order == b_orders.end())
				{
					logger().debug("Missing discretization order for body {}; using 1", bid);
					b_orders[bid] = 1;
					disc_orders[e] = 1;
				}
				else
				{
					disc_orders[e] = order->second;
				}
			}
		}
		else
		{
			logger().error("space/discr_order must be either a number a path or an array");
			throw std::runtime_error("invalid json");
		}

		DiscOrders result;
		result.ordersq = disc_orders;

		if (args["space"]["use_p_ref"])
		{
			refinement::APriori::p_refine(
				mesh,
				args["space"]["advanced"]["B"],
				args["space"]["advanced"]["h1_formula"],
				args["space"]["discr_order"],
				args["space"]["advanced"]["discr_order_max"],
				stats,
				disc_orders);

			logger().info("min p: {} max p: {}", disc_orders.minCoeff(), disc_orders.maxCoeff());
		}

		result.orders = disc_orders;
		return result;
	}

	Eigen::VectorXi resolve_geom_orders(const mesh::Mesh &mesh,
										const Eigen::VectorXi &disc_orders)
	{
		if (mesh.orders().size() <= 0)
		{
			Eigen::VectorXi geom_orders(disc_orders.size());
			geom_orders.setConstant(1);
			return geom_orders;
		}

		return mesh.orders().cast<int>();
	}
} // namespace polyfem::varform

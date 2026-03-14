#include <polyfem/varforms/VarForm.hpp>

#include <polyfem/io/MatrixIO.hpp>

#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/StringUtils.hpp>

namespace polyfem::varform
{
	void VarForm::init(const std::string &formulation, const Units &units, const json &args, const std::string &out_path)
	{
		reset();

		this->units = units;

		if (utils::is_param_valid(args, "root_path"))
			root_path = args["root_path"].get<std::string>();
		else
			root_path = "";

		this->output_path = out_path;
	}

	void VarForm::assign_discr_orders(const json &discr_order, const mesh::Mesh &mesh, Eigen::VectorXi &disc_orders)
	{
		disc_orders.resize(mesh.n_elements());

		if (discr_order.is_number_integer())
		{
			disc_orders.setConstant(discr_order);
		}
		else if (discr_order.is_string())
		{
			const std::string discr_orders_path = utils::resolve_path(discr_order, root_path);
			Eigen::MatrixXi tmp;
			io::read_matrix(discr_orders_path, tmp);
			assert(tmp.size() == disc_orders.size());
			assert(tmp.cols() == 1);
			disc_orders = tmp;
		}
		else if (discr_order.is_array())
		{
			const auto b_discr_orders = discr_order;

			std::map<int, int> b_orders;
			for (size_t i = 0; i < b_discr_orders.size(); ++i)
			{
				assert(b_discr_orders[i]["id"].is_array() || b_discr_orders[i]["id"].is_number_integer());

				const int order = b_discr_orders[i]["order"];
				for (const int id : utils::json_as_array<int>(b_discr_orders[i]["id"]))
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
	}

	std::string VarForm::resolve_output_path(const std::string &path) const
	{
		if (output_path.empty() || path.empty() || std::filesystem::path(path).is_absolute())
		{
			return path;
		}
		return std::filesystem::weakly_canonical(std::filesystem::path(output_path) / path).string();
	}
} // namespace polyfem::varform

#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace polyfem::io
{
	class MshReader
	{
	public:
		static bool load(
			const std::string &path,
			Eigen::MatrixXd &vertices,
			Eigen::MatrixXi &cells,
			std::vector<std::vector<int>> &elements,
			std::vector<std::vector<double>> &weights,
			std::vector<int> &body_ids);

		static bool load(
			const std::string &path,
			Eigen::MatrixXd &vertices,
			Eigen::MatrixXi &cells,
			std::vector<std::vector<int>> &elements,
			std::vector<std::vector<double>> &weights,
			std::vector<int> &body_ids,
			std::vector<std::string> &node_data_name,
			std::vector<std::vector<double>> &node_data);
	};
} // namespace polyfem::io

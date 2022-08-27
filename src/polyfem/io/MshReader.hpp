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
	};
} // namespace polyfem::io

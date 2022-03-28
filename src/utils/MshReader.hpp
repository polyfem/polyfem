#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace polyfem
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
			Eigen::VectorXi &body_ids);
	};
} // namespace polyfem

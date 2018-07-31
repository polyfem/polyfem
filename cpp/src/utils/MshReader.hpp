#pragma once

#include <Eigen/Dense>
#include <string>

namespace polyfem
{
	class MshReader
	{
	public:
		static bool load(const std::string &path, Eigen::MatrixXd &vertices, Eigen::MatrixXi &cells, std::vector<std::vector<int>> &elements);
	};
}
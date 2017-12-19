#pragma once
#include "Navigation3D.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
namespace poly_fem{
	class Mesh3D{
		bool load(Mesh &mesh, const std::string &path);
		bool save(Mesh &mesh, const std::string &path) const;

	};
}
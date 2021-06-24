////////////////////////////////////////////////////////////////////////////////
#include <polyfem/MatrixUtils.hpp>
#include <polyfem/RBFInterpolation.hpp>

#include <catch.hpp>
#include <iostream>
#include <fstream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;

TEST_CASE("interpolation", "[rbf_test]")
{
	const std::string path = POLYFEM_DATA_DIR;
	Eigen::MatrixXd disp, pts;
	read_matrix(path + "/disp.txt", disp);
	read_matrix(path + "/pts.txt", pts);

	RBFInterpolation rbf(disp, pts, "thin_plate", 1e-2);

	const auto vals = rbf.interpolate(pts);

	for (int i = 0; i < pts.rows(); ++i)
	{
		for (int j = 0; j < pts.cols(); ++j)
		{
			REQUIRE(disp(i, j) == Approx(vals(i, j)).margin(1e-9));
		}
	}

	// std::ofstream file("xxx.txt");
	// file << vals;
}

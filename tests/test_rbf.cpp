////////////////////////////////////////////////////////////////////////////////
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/utils/RBFInterpolation.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::io;
using namespace polyfem::utils;

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
			REQUIRE(disp(i, j) == Catch::Approx(vals(i, j)).margin(1e-9));
		}
	}
}

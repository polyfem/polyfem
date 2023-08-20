#include <polyfem/Common.hpp>
#include <polyfem/utils/Interpolation.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace polyfem;
using namespace polyfem::utils;

TEST_CASE("piecewise cubic interpolation", "[interpolation]")
{
	json params = R"(
    {
        "type": "piecewise_cubic",
        "points": [1, 2, 3, 5],
        "values": [3, 2, -1, 1]
    }
    )"_json;

	Eigen::MatrixXd expected_coeffs(3, 4);
	SECTION("constant")
	{
		params["extend"] = "constant";
		// clang-format off
		expected_coeffs <<
            -13,   30,  -21,   70,
             39, -282,  603, -346,
            -20,  249, -990, 1247;
		// clang-format on
		expected_coeffs /= 22.0;
	}
	SECTION("extrapolate")
	{
		params["extend"] = "extrapolate";
		// clang-format off
	    expected_coeffs <<
	        -16,   48,  -55,   92,
             34, -252,  545, -308,
             -9,  135, -616,  853;
		// clang-format on
		expected_coeffs /= 23.0;
	}
	SECTION("repeat_offset")
	{
		params["extend"] = "repeat_offset";
		// clang-format off
	    expected_coeffs <<
	         -1,  -30,   77,   14,
	         31, -222,  461, -242,
	        -15,  192, -781, 1000;
		// clang-format on
		expected_coeffs /= 20.0;
	}

	const std::shared_ptr<Interpolation> interp = Interpolation::build(params);
	REQUIRE(interp != nullptr);
	const std::shared_ptr<PiecewiseCubicInterpolation> piecewise_interp = std::dynamic_pointer_cast<PiecewiseCubicInterpolation>(interp);

	const Eigen::MatrixXd &coeffs = piecewise_interp->coeffs();
	REQUIRE(coeffs.rows() == expected_coeffs.rows());
	REQUIRE(coeffs.cols() == expected_coeffs.cols());

	for (int i = 0; i < coeffs.rows(); ++i)
		for (int j = 0; j < coeffs.cols(); ++j)
			CHECK(coeffs(i, j) == Catch::Approx(expected_coeffs(i, j)));

	const std::vector<double> points = params["points"].get<std::vector<double>>();
	const std::vector<double> values = params["values"].get<std::vector<double>>();

	for (int i = 0; i < points.size(); ++i)
	{
		CAPTURE(points[i]);
		CHECK(interp->eval(points[i]) == Catch::Approx(values[i]));
	}
}

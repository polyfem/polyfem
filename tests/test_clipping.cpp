#include <catch2/catch_all.hpp>

#include <polyfem/utils/ClipperUtils.hpp>
#include <polyfem/utils/GeometryUtils.hpp>

TEST_CASE("Tetrahedra clipping", "[clipping]")
{
	using namespace polyfem::utils;

	const double dz = GENERATE(range(-0.99, 1.0, 0.01));

	Eigen::MatrixXd subject_tet(4, 3);
	subject_tet << 0, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1;

	Eigen::MatrixXd clipping_tet = subject_tet;
	clipping_tet.col(2).array() += dz;

	std::vector<Eigen::MatrixXd> r = TetrahedronClipping::clip(subject_tet, clipping_tet);

	Eigen::MatrixXd expected_tet(4, 3);
	expected_tet = (1 - abs(dz)) * subject_tet;
	expected_tet.col(2).array() += std::max(0.0, dz);

	CAPTURE(dz);
	REQUIRE(r.size() == 1);
	REQUIRE(r[0].rows() == 4);
	REQUIRE(r[0].cols() == 3);
	if (std::abs(dz) > 1e-15)
		CHECK(r[0].colwise().minCoeff().isApprox(expected_tet.colwise().minCoeff()));
	else
		CHECK(r[0].colwise().minCoeff().isZero());
	CHECK(r[0].colwise().maxCoeff().isApprox(expected_tet.colwise().maxCoeff()));
	CHECK(r[0].colwise().sum().isApprox(expected_tet.colwise().sum()));
	CHECK(tetrahedron_volume(r[0]) == Catch::Approx(tetrahedron_volume(expected_tet)));
	// Node order might be different
	for (int i = 0; i < 4; i++)
	{
		int j = 0;
		for (int j = 0; j < 4; j++)
		{
			if (r[0].row(i).isApprox(expected_tet.row(j)))
				break;
		}
		CHECK(j < 4);
	}
}

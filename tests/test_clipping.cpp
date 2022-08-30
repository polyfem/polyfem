#include <catch2/catch.hpp>

#include <polyfem/utils/ClipperUtils.hpp>

#include <iostream>

TEST_CASE("Tetrahedra clipping", "[clipping]")
{
	const double dz = 0.1; // GENERATE(take(100, random(-1.0, 1.0)));

	Eigen::MatrixXd subject_tet(4, 3);
	subject_tet << 0, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1;

	Eigen::MatrixXd clipping_tet = subject_tet;
	clipping_tet.col(2).array() += dz;

	std::vector<Eigen::MatrixXd> r = polyfem::utils::TetrahedronClipping::clip(subject_tet, clipping_tet);

	Eigen::MatrixXd expected_tet(4, 3);
	expected_tet = (1 - dz) * subject_tet;
	if (dz > 0)
		expected_tet.col(2).array() += dz;

	CAPTURE(dz);
	REQUIRE(r.size() == 1);
	std::cout << dz << std::endl;
	std::cout << r[0] << std::endl;
	std::cout << std::endl;
	std::cout << subject_tet << std::endl;
	std::cout << std::endl;
	std::cout << clipping_tet << std::endl;
	std::cout << std::endl;
	std::cout << expected_tet << std::endl;
	REQUIRE(r[0].isApprox(expected_tet));
}

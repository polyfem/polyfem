#include <catch2/catch.hpp>

#include <polyfem/utils/ClipperUtils.hpp>

#include <ipc/utils/eigen_ext.hpp>

namespace
{
	double tet_signed_volume(const Eigen::MatrixXd &T)
	{
		return ipc::cross(T.row(1) - T.row(0), T.row(2) - T.row(0)).dot(T.row(3) - T.row(0));
	}
} // namespace

TEST_CASE("Tetrahedra clipping", "[clipping]")
{
	const double dz = GENERATE(take(100, random(-1.0, 1.0)));

	Eigen::MatrixXd subject_tet(4, 3);
	subject_tet << 0, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1;

	Eigen::MatrixXd clipping_tet = subject_tet;
	clipping_tet.col(2).array() += dz;

	std::vector<Eigen::MatrixXd> r = polyfem::utils::TetrahedronClipping::clip(subject_tet, clipping_tet);

	Eigen::MatrixXd expected_tet(4, 3);
	expected_tet = (1 - abs(dz)) * subject_tet;
	expected_tet.col(2).array() += std::max(0.0, dz);

	CAPTURE(dz);
	REQUIRE(r.size() == 1);
	REQUIRE(r[0].rows() == 4);
	REQUIRE(r[0].cols() == 3);
	CHECK(r[0].colwise().minCoeff().isApprox(expected_tet.colwise().minCoeff()));
	CHECK(r[0].colwise().maxCoeff().isApprox(expected_tet.colwise().maxCoeff()));
	CHECK(r[0].colwise().sum().isApprox(expected_tet.colwise().sum()));
	CHECK(tet_signed_volume(r[0]) == Approx(tet_signed_volume(expected_tet)));
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

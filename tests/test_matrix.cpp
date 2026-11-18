////////////////////////////////////////////////////////////////////////////////
#include <polyfem/utils/MatrixCache.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/autogen/auto_eigs.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

#include <iostream>
#include <cmath>

#include <Eigen/Dense>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;
using namespace polyfem::utils;

TEST_CASE("determinant2", "[matrix]")
{
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat(2, 2);
	mat.setRandom();

	REQUIRE(determinant(mat) == Catch::Approx(mat.determinant()).margin(1e-12));
}

TEST_CASE("determinant3", "[matrix]")
{
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> mat(3, 3);
	mat.setRandom();

	REQUIRE(determinant(mat) == Catch::Approx(mat.determinant()).margin(1e-12));
}

TEST_CASE("eigs2id", "[matrix]")
{
	Eigen::Matrix<double, 2, 2> mat;
	mat.setIdentity();

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> actual_eigs(2);
	autogen::eigs_2d<double>(mat, actual_eigs);

	const auto e0 = Catch::Approx(1).margin(1e-8);
	const auto e1 = Catch::Approx(1).margin(1e-8);

	REQUIRE(actual_eigs(0) == e0);
	REQUIRE(actual_eigs(1) == e1);
}

TEST_CASE("eigs2", "[matrix]")
{
	Eigen::Matrix<double, 2, 2> mat;
	mat << 0.679900336682452, 0.853956838129916,
		0.853956838129916, 1.654714487848889;

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> actual_eigs(2);
	autogen::eigs_2d<double>(mat, actual_eigs);

	const auto e0 = Catch::Approx(0.184043491073894).margin(1e-8);
	const auto e1 = Catch::Approx(2.150571333457448).margin(1e-8);

	REQUIRE(actual_eigs(0) == e0);
	REQUIRE(actual_eigs(1) == e1);
}

TEST_CASE("eigs2autodiff", "[matrix]")
{
	typedef DScalar2<double, Eigen::Matrix<double, 4, 1>, Eigen::Matrix<double, 4, 4>> T;
	typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

	Eigen::Matrix<double, 2, 2> tmp;
	tmp << 0.679900336682452, 0.853956838129916,
		0.853956838129916, 1.654714487848889;

	DiffScalarBase::setVariableCount(4);
	AutoDiffGradMat mat(2, 2);
	for (int i = 0; i < 4; ++i)
		mat(i) = T(i, tmp(i));

	Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> actual_eigs(2);
	autogen::eigs_2d<T>(mat, actual_eigs);

	const auto e0 = Catch::Approx(0.184043491073894).margin(1e-8);
	const auto e1 = Catch::Approx(2.150571333457448).margin(1e-8);

	REQUIRE(actual_eigs(0).getValue() == e0);
	REQUIRE(actual_eigs(1).getValue() == e1);
}

TEST_CASE("eigs3id", "[matrix]")
{
	Eigen::Matrix<double, 3, 3> mat;
	mat.setIdentity();

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> actual_eigs(3);
	autogen::eigs_3d<double>(mat, actual_eigs);

	const auto e0 = Catch::Approx(1).margin(1e-15);
	const auto e1 = Catch::Approx(1).margin(1e-15);
	const auto e2 = Catch::Approx(1).margin(1e-15);

	REQUIRE(actual_eigs(0) == e0);
	REQUIRE(actual_eigs(1) == e1);
	REQUIRE(actual_eigs(2) == e2);
}

TEST_CASE("eigs3", "[matrix]")
{
	Eigen::Matrix<double, 3, 3> mat;
	mat << 0.723799495850613, 0.738301489180688, 0.854652664143603,
		0.738301489180688, 1.868383818516662, 1.880071527849840,
		0.854652664143603, 1.880071527849840, 1.924739710079838;

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> actual_eigs(3);
	autogen::eigs_3d<double>(mat, actual_eigs);

	const auto e0 = Catch::Approx(4.147907219689489).margin(1e-8);
	const auto e1 = Catch::Approx(0.002128421644668).margin(1e-8);
	const auto e2 = Catch::Approx(0.366887383112958).margin(1e-8);

	REQUIRE(actual_eigs(0) == e0);
	REQUIRE(actual_eigs(1) == e1);
	REQUIRE(actual_eigs(2) == e2);
}

TEST_CASE("eigs3autodiff", "[matrix]")
{
	typedef DScalar2<double, Eigen::Matrix<double, 9, 1>, Eigen::Matrix<double, 9, 9>> T;
	typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;
	Eigen::Matrix<double, 3, 3> tmp;
	tmp << 0.723799495850613, 0.738301489180688, 0.854652664143603,
		0.738301489180688, 1.868383818516662, 1.880071527849840,
		0.854652664143603, 1.880071527849840, 1.924739710079838;

	DiffScalarBase::setVariableCount(9);
	AutoDiffGradMat mat(3, 3);
	for (int i = 0; i < 9; ++i)
		mat(i) = T(i, tmp(i));

	Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> actual_eigs(3);
	autogen::eigs_3d<T>(mat, actual_eigs);

	const auto e0 = Catch::Approx(4.147907219689489).margin(1e-8);
	const auto e1 = Catch::Approx(0.002128421644668).margin(1e-8);
	const auto e2 = Catch::Approx(0.366887383112958).margin(1e-8);

	REQUIRE(actual_eigs(0).getValue() == e0);
	REQUIRE(actual_eigs(1).getValue() == e1);
	REQUIRE(actual_eigs(2).getValue() == e2);

	for (int k = 0; k < 3; ++k)
	{
		for (int i = 0; i < 9; ++i)
		{
			REQUIRE(!std::isnan(actual_eigs(k).getGradient()(i)));

			for (int j = 0; j < 9; ++j)
			{
				REQUIRE(!std::isnan(actual_eigs(k).getHessian()(i, j)));
			}
		}
	}
}

TEST_CASE("eigs3idautodiff", "[matrix]")
{
	typedef DScalar2<double, Eigen::Matrix<double, 9, 1>, Eigen::Matrix<double, 9, 9>> T;
	typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;
	Eigen::Matrix<double, 3, 3> tmp;
	tmp.setIdentity();

	DiffScalarBase::setVariableCount(9);
	AutoDiffGradMat mat(3, 3);
	for (int i = 0; i < 9; ++i)
		mat(i) = T(i, tmp(i));

	Eigen::Matrix<T, Eigen::Dynamic, 1, 0, 3, 1> actual_eigs(3);
	autogen::eigs_3d<T>(mat, actual_eigs);

	const auto e0 = Catch::Approx(1).margin(1e-15);
	const auto e1 = Catch::Approx(1).margin(1e-15);
	const auto e2 = Catch::Approx(1).margin(1e-15);

	REQUIRE(actual_eigs(0).getValue() == e0);
	REQUIRE(actual_eigs(1).getValue() == e1);
	REQUIRE(actual_eigs(2).getValue() == e2);

	for (int k = 0; k < 3; ++k)
	{
		for (int i = 0; i < 9; ++i)
		{
			REQUIRE(!std::isnan(actual_eigs(k).getGradient()(i)));

			for (int j = 0; j < 9; ++j)
			{
				REQUIRE(!std::isnan(actual_eigs(k).getHessian()(i, j)));
			}
		}
	}
}

TEST_CASE("cache", "[matrix]")
{
	SparseMatrixCache cache(10);
	cache.add_value(0, 0, 0, 1);
	cache.add_value(0, 0, 1, 2);
	cache.add_value(0, 9, 4, 3);
	cache.add_value(0, 9, 4, 3);
	cache.add_value(0, 9, 9, 4);

	const auto tmp = cache.get_matrix();

	REQUIRE(tmp.coeff(0, 0) == 1);
	REQUIRE(tmp.coeff(0, 1) == 2);
	REQUIRE(tmp.coeff(9, 4) == 6);
	REQUIRE(tmp.coeff(9, 9) == 4);

	///////////////////
	SparseMatrixCache cache1(10);
	cache1.add_value(0, 0, 0, 1);
	cache1.add_value(0, 0, 1, 2);
	cache1.prune();
	cache1.add_value(0, 9, 4, 3);
	cache1.add_value(0, 9, 4, 3);
	cache1.add_value(0, 9, 9, 4);

	const auto tmp1 = cache1.get_matrix();

	REQUIRE(tmp1.coeff(0, 0) == 1);
	REQUIRE(tmp1.coeff(0, 1) == 2);
	REQUIRE(tmp1.coeff(9, 4) == 6);
	REQUIRE(tmp1.coeff(9, 9) == 4);

	///////////////////
	SparseMatrixCache cache2(cache1);
	cache2.add_value(0, 0, 0, 1);
	cache2.add_value(0, 0, 1, 2);
	cache2.prune();
	cache2.add_value(0, 9, 4, 3);
	cache2.add_value(0, 9, 4, 3);
	cache2.add_value(0, 9, 9, 4);

	const auto tmp2 = cache2.get_matrix();

	REQUIRE(tmp2.coeff(0, 0) == 1);
	REQUIRE(tmp2.coeff(0, 1) == 2);
	REQUIRE(tmp2.coeff(9, 4) == 6);
	REQUIRE(tmp2.coeff(9, 9) == 4);
}

TEST_CASE("cache compressed export", "[matrix]")
{
	SparseMatrixCache cache(4);
	const auto add = [](SparseMatrixCache &target, const double scale) {
		target.add_value(0, 0, 0, scale);
		target.add_value(0, 2, 0, 2 * scale);
		target.add_value(0, 2, 0, -2 * scale);
		target.add_value(1, 3, 2, 3 * scale);
		target.add_value(1, 1, 3, -4 * scale);
	};
	add(cache, 1);
	const StiffnessMatrix first = cache.get_matrix();
	REQUIRE(first.nonZeros() == 4);

	const auto check_pattern = [&](const StiffnessMatrix &actual) {
		REQUIRE(actual.isCompressed());
		REQUIRE(actual.rows() == first.rows());
		REQUIRE(actual.cols() == first.cols());
		REQUIRE(actual.nonZeros() == first.nonZeros());
		for (int i = 0; i <= first.outerSize(); ++i)
			REQUIRE(actual.outerIndexPtr()[i] == first.outerIndexPtr()[i]);
		for (int i = 0; i < first.nonZeros(); ++i)
			REQUIRE(actual.innerIndexPtr()[i] == first.innerIndexPtr()[i]);
	};

	// Reuse the mapped structure, including an explicitly stored zero and empty column.
	add(cache, -2);
	const StiffnessMatrix second = cache.get_matrix();
	check_pattern(second);
	REQUIRE((second + 2 * first).norm() == 0);
	REQUIRE((cache.mat() - second).norm() == 0);
	REQUIRE(first.coeff(0, 0) == 1);

	// A worker shares the parent's index mapping but owns its coefficients.
	SparseMatrixCache worker(cache);
	add(worker, 3);
	const StiffnessMatrix worker_matrix = worker.get_matrix();
	check_pattern(worker_matrix);
	REQUIRE((worker_matrix - 3 * first).norm() == 0);

	// Export clears the value buffer while retaining all structural entries.
	const StiffnessMatrix cleared = cache.get_matrix();
	check_pattern(cleared);
	REQUIRE(cleared.norm() == 0);
	REQUIRE(second.coeff(0, 0) == -2);
	REQUIRE(worker_matrix.coeff(0, 0) == 3);

	SparseMatrixCache empty(4);
	REQUIRE(empty.get_matrix().nonZeros() == 0);
	const StiffnessMatrix empty_cached = empty.get_matrix();
	REQUIRE(empty_cached.isCompressed());
	REQUIRE(empty_cached.rows() == 4);
	REQUIRE(empty_cached.cols() == 4);
	REQUIRE(empty_cached.nonZeros() == 0);
}
